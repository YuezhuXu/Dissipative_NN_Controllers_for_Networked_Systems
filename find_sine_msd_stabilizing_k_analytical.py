import argparse

import numpy as np

from envs.coupled_msd_env import CoupledMSDEnv


def synthesize_gains(m, d, k, q_max, v_max, umax):
    if q_max <= 0 or v_max < 0 or umax <= 0:
        raise ValueError("q_max and umax must be positive; v_max must be nonnegative.")
    if np.any(k * q_max >= umax):
        raise ValueError(
            "umax must exceed k_i * q_max for every subsystem."
        )

    a = m * q_max
    b = 2.0 * m * v_max
    c = k * q_max - d * v_max - umax
    omega = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
    if np.any(2.0 * m * omega <= d):
        raise ValueError(
            "input bounds are too small for positive added damping."
        )

    position_gain = -(k + m * omega**2)
    velocity_gain = -(2.0 * m * omega - d)
    return np.column_stack(
        (position_gain, velocity_gain, np.zeros_like(position_gain))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--m", default="default")
    parser.add_argument("--d", default="default")
    parser.add_argument("--k", default="default")
    parser.add_argument("--q_max", type=float, default=2.4)
    parser.add_argument("--v_max", type=float, default=0.4)
    parser.add_argument("--umax", type=float, default=12.0)
    parser.add_argument("--output", default="sine_msd_stabilizing_k.npz")
    args = parser.parse_args()

    env = CoupledMSDEnv(
        env_type="sine",
        n=args.N,
        m=args.m,
        d=args.d,
        k=args.k,
    )
    K = synthesize_gains(
        env.m, env.d, env.k, args.q_max, args.v_max, args.umax
    )
    np.savez(args.output, K=K)

    for i, gain in enumerate(K, start=1):
        print(f"K_{i} = {gain}")
    print(f"Saved K to {args.output}")


if __name__ == "__main__":
    main()
