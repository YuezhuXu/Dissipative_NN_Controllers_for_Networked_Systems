import argparse

import numpy as np

from envs.coupled_msd_env import CoupledMSDEnv


def compute_qsr(env, q_max):
    alpha = np.sin(q_max) / q_max
    n = env.n
    P = np.repeat(
        np.array([[[0.9, 0.1], [0.1, 0.1]]]), n, axis=0
    )
    Q = np.zeros((n, 2, 2))
    S = np.zeros((n, 2, 1))
    R = 1e-3 * np.ones((n, 1, 1))

    for i, (m, d, k) in enumerate(zip(env.m, env.d, env.k)):
        B = np.array([[0.0], [1.0 / m]])
        endpoint_derivatives = []
        for phi in (alpha, 1.0):
            A = np.array([[0.0, 1.0], [k * phi / m, -d / m]])
            endpoint_derivatives.append(
                0.5 * (P[i] @ A + A.T @ P[i])
            )
        center = 0.5 * sum(endpoint_derivatives)
        radius = np.linalg.norm(
            0.5 * (endpoint_derivatives[1] - endpoint_derivatives[0]), 2
        )
        Q[i] = center + radius * np.eye(2)
        S[i] = 0.5 * P[i] @ B
    return Q, S, R, P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--m", default="default")
    parser.add_argument("--d", default="default")
    parser.add_argument("--k", default="default")
    parser.add_argument("--q_max", type=float, default=2.4)
    parser.add_argument("--output", default="sine_msd_qsr.npz")
    args = parser.parse_args()

    if not 0.0 < args.q_max < np.pi:
        raise ValueError("q_max must satisfy 0 < q_max < pi.")
    env = CoupledMSDEnv(
        env_type="sine",
        n=args.N,
        m=args.m,
        d=args.d,
        k=args.k,
    )
    Q, S, R, P = compute_qsr(env, args.q_max)
    np.savez(args.output, Q=Q, S=S, R=R, P=P)

    for i in range(args.N):
        print(f"Subsystem {i + 1}")
        print("Q =")
        print(Q[i])
        print("S =")
        print(S[i])
        print("R =")
        print(R[i])
        print("P =")
        print(P[i])
    print(f"Saved Q, S, R, P to {args.output}")


if __name__ == "__main__":
    main()
