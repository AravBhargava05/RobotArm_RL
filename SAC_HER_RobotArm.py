import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import imageio.v2 as imageio

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_util import make_vec_env


@dataclass
class Config:
    env_ids: Tuple[str, ...] = ("FetchPickAndPlace-v4", "FetchPickAndPlace-v3")
    seed: int = 0

    # Training
    n_envs: int = 4
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.05
    buffer_size: int = 1_000_000

    # Fetch is ~50-step horizon; HER needs complete episodes before sampling
    max_steps: int = 50
    learning_starts_mult: int = 2  # learning_starts = max_steps * n_envs * this

    # HER
    n_sampled_goal: int = 4
    goal_selection_strategy: GoalSelectionStrategy = GoalSelectionStrategy.FUTURE

    # Video
    videos_per_checkpoint: int = 20
    video_dir: str = "pickplace/videos"
    model_dir: str = "pickplace/models"
    fps: int = 30
    pause_frames_between_eps: int = 8


def pick_env_id(env_ids: Tuple[str, ...], seed: int) -> str:
    last_err = None
    for env_id in env_ids:
        try:
            env = gym.make(env_id)
            env.reset(seed=seed)
            env.close()
            return env_id
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not create any env from {env_ids}. Last error: {last_err}")


def make_video_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def record_rollouts_mp4(
    model: SAC,
    env_id: str,
    out_path: str,
    seed: int,
    n_episodes: int,
    max_steps: int,
    fps: int,
    pause_frames: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = make_video_env(env_id, seed=seed)

    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            done = False
            t = 0

            while not done and t < max_steps:
                writer.append_data(env.render())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                t += 1

            if pause_frames > 0:
                last = env.render()
                for _ in range(pause_frames):
                    writer.append_data(last)
    finally:
        writer.close()
        env.close()


def main():
    cfg = Config()
    os.makedirs(cfg.video_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    env_id = pick_env_id(cfg.env_ids, seed=cfg.seed)
    print(f"Using env: {env_id}")

    # Vectorized training env; make_vec_env already uses Monitor internally
    vec_env = make_vec_env(env_id, n_envs=cfg.n_envs, seed=cfg.seed)

    learning_starts = cfg.max_steps * cfg.n_envs * cfg.learning_starts_mult

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        buffer_size=cfg.buffer_size,
        learning_starts=learning_starts,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=cfg.n_sampled_goal,
            goal_selection_strategy=cfg.goal_selection_strategy,
        ),
        verbose=1,
        seed=cfg.seed,
    )

    pre_path = os.path.join(cfg.video_dir, f"{env_id}_pre_20.mp4")
    print(f"Recording pre-training -> {pre_path}")
    record_rollouts_mp4(
        model, env_id, pre_path,
        seed=cfg.seed + 123,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    half = cfg.total_timesteps // 2
    print(f"Training first half: {half} steps (learning_starts={learning_starts})")
    model.learn(total_timesteps=half, reset_num_timesteps=True)

    mid_path = os.path.join(cfg.video_dir, f"{env_id}_mid_20.mp4")
    print(f"Recording mid-training -> {mid_path}")
    record_rollouts_mp4(
        model, env_id, mid_path,
        seed=cfg.seed + 456,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    rest = cfg.total_timesteps - half
    print(f"Training second half: {rest} steps")
    model.learn(total_timesteps=rest, reset_num_timesteps=False)

    end_path = os.path.join(cfg.video_dir, f"{env_id}_end_20.mp4")
    print(f"Recording final -> {end_path}")
    record_rollouts_mp4(
        model, env_id, end_path,
        seed=cfg.seed + 789,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    model_path = os.path.join(cfg.model_dir, f"sac_her_{env_id}.zip")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
