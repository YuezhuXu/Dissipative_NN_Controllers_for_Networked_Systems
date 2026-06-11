import argparse

import cvxpy as cp
import numpy as np

from envs.coupled_msd_env import make_graph


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


def controller_matrix(local_gains, H):
    n = len(local_gains)
    K_y = cp.bmat(
        [
            [
                local_gains[i][:, :2] if i == j else np.zeros((1, 2))
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    K_z = cp.diag(cp.hstack([gain[0, 2] for gain in local_gains]))
    return K_y + (np.eye(n) + K_z) @ H


def solve_stabilizing_k(Q_blocks, S_blocks, R_blocks, H, gain_bound):
    n = len(Q_blocks)
    Q = block_diag(Q_blocks)
    S = block_diag(S_blocks)
    R = np.asarray(block_diag(R_blocks).value)

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (R + R.T))
    if eigenvalues.min() < -1e-9:
        raise ValueError("R must be positive semidefinite.")
    R_sqrt = (
        eigenvectors
        @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        @ eigenvectors.T
    )

    local_gains = [cp.Variable((1, 3)) for _ in range(n)]
    X = controller_matrix(local_gains, H)
    margin = cp.Variable(nonneg=True)
    top_left = Q + S @ X + X.T @ S.T + margin * np.eye(2 * n)
    lmi = cp.bmat(
        [[top_left, X.T @ R_sqrt], [R_sqrt @ X, -np.eye(n)]]
    )
    constraints = [
        lmi << 0,
        *[cp.norm(gain, 2) <= gain_bound for gain in local_gains],
    ]

    problem = cp.Problem(cp.Maximize(margin), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Controller synthesis failed: {problem.status}")

    return np.vstack([gain.value for gain in local_gains]), float(margin.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsr", default="sine_msd_qsr.npz")
    parser.add_argument(
        "--graph",
        choices=["none", "chain", "ring", "complete", "star"],
        default="complete",
    )
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--gain_bound", type=float, default=10.0)
    parser.add_argument("--output", default="sine_msd_stabilizing_k_qsr.npz")
    args = parser.parse_args()

    qsr = np.load(args.qsr)
    Q, S, R = qsr["Q"], qsr["S"], qsr["R"]
    n = len(Q)
    H = interconnection_matrix(make_graph(n, args.graph, args.coupling))
    K, margin = solve_stabilizing_k(Q, S, R, H, args.gain_bound)
    np.savez(args.output, K=K)

    for i, gain in enumerate(K, start=1):
        print(f"K_{i} = {gain}")
    print(f"Certificate margin: {margin:.6e}")
    print(f"Saved K to {args.output}")


if __name__ == "__main__":
    main()
