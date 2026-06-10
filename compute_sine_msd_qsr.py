import argparse

import cvxpy as cp
import numpy as np

from envs.coupled_msd_env import CoupledMSDEnv, make_graph


def block_diag(blocks):
    n = len(blocks)
    return cp.bmat(
        [
            [
                blocks[i]
                if i == j
                else np.zeros((blocks[i].shape[0], blocks[j].shape[1]))
                for j in range(n)
            ]
            for i in range(n)
        ]
    )


def interconnection_matrix(adjacency):
    n = adjacency.shape[0]
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    H = np.zeros((n, 2 * n))
    H[:, 0::2] = -laplacian
    return H


def compute_qsr(env, q_max, omega):
    alpha = np.sin(q_max) / q_max
    n = env.n
    P_variables = []
    Q_variables = []
    S_expressions = []
    constraints = []

    for m, d, k in zip(env.m, env.d, env.k):
        B = np.array([[0.0], [1.0 / m]])
        P = cp.Variable((2, 2), symmetric=True)
        Q = cp.Variable((2, 2), symmetric=True)
        constraints.extend([P >> 1e-3 * np.eye(2), cp.trace(P) == 1.0])
        for phi in (alpha, 1.0):
            A = np.array([[0.0, 1.0], [k * phi / m, -d / m]])
            constraints.append(
                Q - 0.5 * (P @ A + A.T @ P) >> 0
            )
        P_variables.append(P)
        Q_variables.append(Q)
        S_expressions.append(0.5 * P @ B)

    reference_K = np.column_stack(
        (
            -(env.k + env.m * omega**2),
            -(2.0 * env.m * omega - env.d),
            np.zeros(n),
        )
    )
    H = interconnection_matrix(env.A)
    K_y = np.zeros((n, 2 * n))
    for i in range(n):
        K_y[i, 2 * i : 2 * i + 2] = reference_K[i, :2]
    X = K_y + H

    margin = cp.Variable(nonneg=True)
    Q = block_diag(Q_variables)
    S = block_diag(S_expressions)
    constraints.append(
        Q + S @ X + X.T @ S.T + margin * np.eye(2 * n) << 0
    )
    problem = cp.Problem(cp.Maximize(margin), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"QSR fitting failed: {problem.status}")

    P = np.stack([value.value for value in P_variables])
    Q = np.stack([value.value for value in Q_variables])
    S = np.stack([value.value for value in S_expressions])
    R = np.zeros((n, 1, 1))
    return Q, S, R, P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument(
        "--graph",
        choices=["none", "chain", "ring", "complete", "star"],
        default="complete",
    )
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--m", default="default")
    parser.add_argument("--d", default="default")
    parser.add_argument("--k", default="default")
    parser.add_argument("--q_max", type=float, default=2.4)
    parser.add_argument("--omega", type=float, default=2.0)
    parser.add_argument("--output", default="sine_msd_qsr.npz")
    args = parser.parse_args()

    if not 0.0 < args.q_max < np.pi:
        raise ValueError("q_max must satisfy 0 < q_max < pi.")
    if args.omega <= 0:
        raise ValueError("omega must be positive.")

    env = CoupledMSDEnv(
        env_type="sine",
        n=args.N,
        graph=args.graph,
        coupling=args.coupling,
        m=args.m,
        d=args.d,
        k=args.k,
    )
    Q, S, R, P = compute_qsr(env, args.q_max, args.omega)
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
