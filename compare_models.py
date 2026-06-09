import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from envs.coupled_msd_env import CoupledMSDEnv
from residual_policy import load_model


class FixedLinearPolicy:
    def __init__(self, env, gain, action_scale):
        self.n = env.n
        self.adjacency = env.A
        self.degree = env.degree
        self.gain = np.asarray(gain, dtype=np.float32)
        self.action_scale = action_scale
        self.umax = env.umax

    def predict(self, observation, deterministic=True):
        q = observation[:self.n]
        v = observation[self.n:]
        z = self.adjacency @ q - self.degree * q
        features = np.stack((q, v, z), axis=1)
        raw_action = features @ self.gain
        physical_action = self.action_scale * np.tanh(
            raw_action / self.action_scale
        )
        return physical_action / self.umax, None


def parse_gain(value):
    gain = [float(item.strip()) for item in value.split(",")]
    if len(gain) != 3:
        raise ValueError("linear_gain must contain gains for [q_i, v_i, z_i].")
    return gain


def run(model, env, seed):
    obs, _ = env.reset(seed=seed)
    states = [obs.copy()]
    controls = []
    ret = 0.0
    info = {}
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs.copy())
        controls.append(info["u"].copy())
        ret += reward
        if terminated or truncated:
            break
    return {
        "return": ret,
        "final_state_norm": info["state_norm"],
        "final_action_norm": info["action_norm"],
        "states": np.asarray(states),
        "controls": np.asarray(controls),
    }


def save_plot(path, ylabel, first, second, label_a, label_b):
    plt.figure()
    plt.plot(first, label=label_a)
    plt.plot(second, label=label_b)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_comparison(rollouts_a, rollouts_b, save_dir, label_a, label_b, n):
    states_a = np.stack([rollout["states"] for rollout in rollouts_a])
    states_b = np.stack([rollout["states"] for rollout in rollouts_b])
    controls_a = np.stack([rollout["controls"] for rollout in rollouts_a])
    controls_b = np.stack([rollout["controls"] for rollout in rollouts_b])

    save_plot(
        os.path.join(save_dir, "mean_state_norm_comparison.png"),
        "Mean state norm",
        np.linalg.norm(states_a, axis=2).mean(axis=0),
        np.linalg.norm(states_b, axis=2).mean(axis=0),
        label_a,
        label_b,
    )
    save_plot(
        os.path.join(save_dir, "mean_action_norm_comparison.png"),
        "Mean control norm",
        np.linalg.norm(controls_a, axis=2).mean(axis=0),
        np.linalg.norm(controls_b, axis=2).mean(axis=0),
        label_a,
        label_b,
    )

    for filename, data_a, data_b, ylabel in (
        ("positions_ep0.png", states_a[0, :, :n], states_b[0, :, :n], "Position"),
        ("velocities_ep0.png", states_a[0, :, n:], states_b[0, :, n:], "Velocity"),
        ("controls_ep0.png", controls_a[0], controls_b[0], "Control"),
    ):
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(7, 1.7 * n))
        for i, axis in enumerate(axes):
            axis.plot(data_a[:, i], label=label_a)
            axis.plot(data_b[:, i], label=label_b)
            axis.set_ylabel(f"{ylabel} {i + 1}")
            axis.grid(alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    model_a = parser.add_mutually_exclusive_group(required=True)
    model_a.add_argument("--model_a")
    model_a.add_argument("--fixed_linear_a", action="store_true")
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--env", choices=["sine", "msd"], default="sine")
    parser.add_argument("--save_dir", default="compare_two_models")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--label_a", default="model_a")
    parser.add_argument("--label_b", default="model_b")
    parser.add_argument("--linear_gain", default="-4.8,-1.2,0.0")
    parser.add_argument("--action_scale", type=float)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env_a = CoupledMSDEnv(args.env)
    env_b = CoupledMSDEnv(args.env)
    if args.fixed_linear_a:
        action_scale = env_a.umax if args.action_scale is None else args.action_scale
        model_a = FixedLinearPolicy(env_a, parse_gain(args.linear_gain), action_scale)
    else:
        model_a = load_model(args.model_a, env_a)
    model_b = load_model(args.model_b, env_b)

    rows = []
    rollouts_a = []
    rollouts_b = []
    for i in range(args.episodes):
        seed = 123 + i
        rollout_a = run(model_a, env_a, seed)
        rollout_b = run(model_b, env_b, seed)
        rollouts_a.append(rollout_a)
        rollouts_b.append(rollout_b)
        rows.append({
            "model": args.label_a,
            "episode": i,
            "return": rollout_a["return"],
            "final_state_norm": rollout_a["final_state_norm"],
            "final_action_norm": rollout_a["final_action_norm"],
        })
        rows.append({
            "model": args.label_b,
            "episode": i,
            "return": rollout_b["return"],
            "final_state_norm": rollout_b["final_state_norm"],
            "final_action_norm": rollout_b["final_action_norm"],
        })

    with open(os.path.join(args.save_dir, "comparison_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_comparison(
        rollouts_a,
        rollouts_b,
        args.save_dir,
        args.label_a,
        args.label_b,
        env_a.n,
    )


if __name__ == "__main__":
    main()
