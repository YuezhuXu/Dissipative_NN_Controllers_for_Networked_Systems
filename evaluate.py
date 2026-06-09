import argparse

from envs.coupled_msd_env import CoupledMSDEnv
from residual_policy import load_model


def rollout(model, env, seed=123):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    final_info = {}
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_info = info
        if terminated or truncated:
            break
    return total_reward, final_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--env", choices=["sine", "msd"], default="sine")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = CoupledMSDEnv(args.env)
    model = load_model(args.model_path, env)
    reward, info = rollout(model, env, args.seed)
    print(f"return={reward:.6f}")
    print(f"final_state_norm={info['state_norm']:.6f}")
    print(f"final_action_norm={info['action_norm']:.6f}")


if __name__ == "__main__":
    main()
