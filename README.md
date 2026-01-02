# RobotArm — Fetch Pick & Place (SAC + HER)

Train a simulated robotic arm to **pick up a block and place it at a target** using **Soft Actor-Critic (SAC)** with **Hindsight Experience Replay (HER)** in the Gymnasium Robotics Fetch environment.

This is the classic “sparse reward” manipulation problem: most episodes fail early, so learning is hard without the right algorithmic tricks.

---

## What this project demonstrates

- **Goal-conditioned RL**: the policy is trained to reach a *desired goal* (target position), not just maximize a generic reward.
- **Off-policy continuous control**: SAC handles continuous actions smoothly (unlike DQN-style discrete control).
- **Sparse-reward learning with HER**: the replay buffer relabels failed attempts into “successful” training examples by swapping the goal after the fact.
- **Robotics relevance**: this is a clean demo of how modern RL can learn manipulation behaviors without hand-coded trajectories.

---

## Environment

- **Task:** `FetchPickAndPlace-v4` (falls back to `-v3` if needed)
- **Episode horizon:** 50 steps
- **Observation:** a dictionary with:
  - `observation` (robot + object state)
  - `achieved_goal` (where the object currently is)
  - `desired_goal` (where the object should end up)

---

## Method

### SAC (Soft Actor-Critic)
SAC learns a stochastic policy for continuous control and is generally stable + sample-efficient.

### HER (Hindsight Experience Replay)
For goal-based tasks, HER improves learning dramatically by replaying a failed episode as if the goal had been the thing you actually achieved.

Example:  
If the robot drops the block at position A instead of the target position B, HER treats that episode as a success for goal = A during training.

That turns “wasted” rollouts into useful data.

---

## Results (my run)

Final evaluation output:

- `success_rate = 1.000`
- `return_mean  = -9.4`
- `ep_len_mean  = 50.0`

Notes:
- In Fetch tasks, the environment’s **`is_success`** signal is the real metric.
- The mean return being negative is normal: you often get per-step penalties until success.

---

## Videos

I recorded two rollouts:
- **Mid-training** (≈ 1,000,000 steps): shows partially-learned behavior (more hesitation / less consistent grasp + placement)
- **Final** (end of training): near-perfect pick-and-place behavior

Add your file names here once you upload them:

- `videos/pickplace_mid_1M.mp4`
- `videos/pickplace_final.mp4`

---

## How to run

### Install
You need Gymnasium Robotics + SB3:

```bash
pip install "gymnasium-robotics[mujoco]" stable-baselines3 imageio
