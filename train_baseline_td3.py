import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.torch_utils import policy_within_training_step

from envs.coupled_msd_env import CoupledMSDEnv
from residual_policy import TD3Model, build_td3, save_model


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def parse_list(value, cast=float):
    return [cast(item.strip()) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", choices=["sine", "msd"], default="sine")
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--graph", choices=["none", "chain", "ring", "complete", "star"], default="complete")
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--m", default="default")
    parser.add_argument("--d", default="default")
    parser.add_argument("--k", default="default")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--episode_len", type=int, default=250)
    parser.add_argument("--umax", type=float, default=12.0)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--v_weight", type=float, default=0.3)
    parser.add_argument("--u_weight", type=float, default=0.03)
    parser.add_argument("--terminal_weight", type=float, default=0.0)
    parser.add_argument("--q_init_low", type=float, default=-2.4)
    parser.add_argument("--q_init_high", type=float, default=2.4)
    parser.add_argument("--v_init_low", type=float, default=-0.4)
    parser.add_argument("--v_init_high", type=float, default=0.4)

    parser.add_argument("--actor_mode", choices=["residual", "linear"], default="residual")
    parser.add_argument("--linear_init")
    parser.add_argument(
        "--linear_init_path", default="sine_msd_stabilizing_k.npz"
    )
    parser.add_argument("--action_scale", type=float, default=12.0)
    parser.add_argument("--actor_hidden_layers", default="64,64,64")
    parser.add_argument("--critic_hidden_layers", default="128,128,128")
    parser.add_argument("--actor_activation", choices=ACTIVATIONS, default="elu")
    parser.add_argument("--critic_activation", choices=ACTIVATIONS, default="elu")

    parser.add_argument("--actor_lr", type=float, default=5e-6)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--early_stop_final_norm", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--best_patience", type=int, default=80)
    parser.add_argument("--best_min_delta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration_noise", type=float, default=0.05)
    parser.add_argument("--policy_noise", type=float, default=0.05)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_delay", type=int, default=20)
    parser.add_argument("--save_dir", default="results_sine_inverted_complete")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_env(args):
    return CoupledMSDEnv(
        env_type=args.env_type,
        n=args.N,
        graph=args.graph,
        coupling=args.coupling,
        m=args.m,
        d=args.d,
        k=args.k,
        dt=args.dt,
        episode_len=args.episode_len,
        umax=args.umax,
        q_weight=args.q_weight,
        v_weight=args.v_weight,
        u_weight=args.u_weight,
        terminal_weight=args.terminal_weight,
        q_init_low=args.q_init_low,
        q_init_high=args.q_init_high,
        v_init_low=args.v_init_low,
        v_init_high=args.v_init_high,
    )


def evaluate(model, env, episodes):
    rewards = []
    lengths = []
    final_norms = []
    for _ in range(episodes):
        obs, _ = env.reset()
        reward_sum = 0.0
        length = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            length += 1
            if terminated or truncated:
                break
        rewards.append(reward_sum)
        lengths.append(length)
        final_norms.append(info["state_norm"])
    return np.asarray(rewards), np.asarray(lengths), np.asarray(final_norms)


def print_evaluation(
    step,
    rewards,
    lengths,
    final_norms,
    best_reward,
    no_progress,
    stats,
    updates,
):
    print(
        f"Eval num_timesteps={step}, episode_reward="
        f"{rewards.mean():.2f} +/- {rewards.std():.2f}"
    )
    print(f"Episode length: {lengths.mean():.2f} +/- {lengths.std():.2f}")
    print(f"Mean final state norm: {final_norms.mean():.5f}")
    print("---------------------------------")
    print("| eval/              |          |")
    print(f"|    mean_ep_length  | {lengths.mean():<8.3g} |")
    print(f"|    mean_final_norm | {final_norms.mean():<8.3g} |")
    print(f"|    mean_reward     | {rewards.mean():<8.3g} |")
    print(f"|    best_reward     | {best_reward:<8.3g} |")
    print(f"|    no_progress     | {no_progress:<8d} |")
    print("| time/              |          |")
    print(f"|    total_timesteps | {step:<8d} |")
    if stats is not None:
        print("| train/             |          |")
        print(f"|    actor_loss      | {stats.actor_loss:<8.3g} |")
        critic_loss = stats.critic1_loss + stats.critic2_loss
        print(f"|    critic_loss     | {critic_loss:<8.3g} |")
        print(f"|    n_updates       | {updates:<8d} |")
    print("---------------------------------")


def main():
    args = parse_args()
    if args.action_scale <= 0 or args.action_scale > args.umax:
        raise ValueError("action_scale must satisfy 0 < action_scale <= umax.")

    if args.linear_init is not None:
        linear_init = parse_list(args.linear_init)
    else:
        linear_init = np.load(args.linear_init_path)["K"].tolist()
    linear_init_array = np.asarray(linear_init)
    if linear_init_array.shape not in {(3,), (args.N, 3)}:
        raise ValueError(
            "linear_init must have shape (3,) or (N, 3) for "
            "[q_i, v_i, z_i]."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    env = make_env(args)
    eval_env = make_env(args)
    eval_env.reset(seed=args.seed + 1)
    config = {
        "n_subsystems": args.N,
        "adjacency": env.A.tolist(),
        "linear_init": linear_init,
        "action_scale": args.action_scale,
        "umax": args.umax,
        "actor_mode": args.actor_mode,
        "actor_hidden_layers": parse_list(args.actor_hidden_layers, int),
        "critic_hidden_layers": parse_list(args.critic_hidden_layers, int),
        "actor_activation": args.actor_activation,
        "critic_activation": args.critic_activation,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "exploration_noise": args.exploration_noise,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_delay": args.policy_delay,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    algorithm = build_td3(env, config, device)
    model = TD3Model(algorithm)
    buffer = ReplayBuffer(args.buffer_size)
    collector_env = DummyVectorEnv([lambda: env])
    logging.getLogger("tianshou.data.collector").setLevel(logging.CRITICAL)
    collector = Collector(
        algorithm, collector_env, buffer, exploration_noise=True
    )
    collector.reset(gym_reset_kwargs={"seed": args.seed})

    best_model_reward = -np.inf
    stopping_best_reward = -np.inf
    no_improvement_count = 0
    final_norm_count = 0
    updates = 0
    stats = None

    for step in range(1, args.total_steps + 1):
        algorithm.train()
        with policy_within_training_step(algorithm.policy):
            collector.collect(n_step=1, random=step <= args.start_steps)
            if step > args.start_steps and len(buffer) >= args.batch_size:
                stats = algorithm.update(
                    buffer=buffer,
                    sample_size=args.batch_size,
                )
                updates += 1

        if step % args.eval_every != 0:
            continue

        rewards, lengths, final_norms = evaluate(
            model, eval_env, args.eval_episodes
        )
        mean_reward = float(rewards.mean())
        mean_final_norm = float(final_norms.mean())
        current_best_reward = max(best_model_reward, mean_reward)

        if mean_reward > stopping_best_reward + args.best_min_delta:
            stopping_best_reward = mean_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print_evaluation(
            step,
            rewards,
            lengths,
            final_norms,
            current_best_reward,
            no_improvement_count,
            stats,
            updates,
        )

        if mean_reward > best_model_reward:
            best_model_reward = mean_reward
            save_model(
                os.path.join(args.save_dir, "best_model.pth"),
                algorithm,
                config,
            )

        if mean_final_norm <= args.early_stop_final_norm:
            final_norm_count += 1
        else:
            final_norm_count = 0

        if final_norm_count >= args.early_stop_patience:
            print(
                f"Stopping: mean final norm <= {args.early_stop_final_norm} "
                f"for {args.early_stop_patience} consecutive evaluations."
            )
            break
        if no_improvement_count >= args.best_patience:
            print(
                f"Stopping: mean reward did not improve by "
                f"{args.best_min_delta} for {args.best_patience} "
                "consecutive evaluations."
            )
            break

    save_model(
        os.path.join(args.save_dir, "final_model.pth"),
        algorithm,
        config,
    )


if __name__ == "__main__":
    main()
