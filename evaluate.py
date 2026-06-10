import argparse

import matplotlib.pyplot as plt
import numpy as np

from compare_models import FixedLinearPolicy, parse_gain, run
from envs.coupled_msd_env import CoupledMSDEnv
from residual_policy import load_model


def plot_rollout(path, rollout, n):
    states = rollout["states"]
    controls = rollout["controls"]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(states[:, :n])
    axes[1].plot(states[:, n:])
    axes[2].plot(controls)
    axes[3].plot(np.linalg.norm(states, axis=1), color="black")
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[2].set_ylabel("Control")
    axes[3].set_ylabel("State norm")
    axes[3].set_xlabel("Time step")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--fixed_linear", action="store_true")
    parser.add_argument("--linear_gain")
    parser.add_argument(
        "--linear_gain_path", default="sine_msd_stabilizing_k.npz"
    )
    parser.add_argument("--action_scale", type=float)
    parser.add_argument("--plot")
    parser.add_argument("--env", choices=["sine", "msd"], default="sine")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = CoupledMSDEnv(args.env)
    if args.fixed_linear:
        if args.linear_gain is not None:
            gain = parse_gain(args.linear_gain)
        else:
            gain = np.load(args.linear_gain_path)["K"]
        action_scale = env.umax if args.action_scale is None else args.action_scale
        model = FixedLinearPolicy(env, gain, action_scale)
    elif args.model_path:
        model = load_model(args.model_path, env)
    else:
        parser.error("provide model_path or --fixed_linear")

    result = run(model, env, args.seed)
    print(f"return={result['return']:.6f}")
    print(f"initial_state_norm={np.linalg.norm(result['states'][0]):.6f}")
    print(f"final_state_norm={result['final_state_norm']:.6f}")
    print(f"final_action_norm={result['final_action_norm']:.6f}")
    if args.plot:
        plot_rollout(args.plot, result, env.n)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
