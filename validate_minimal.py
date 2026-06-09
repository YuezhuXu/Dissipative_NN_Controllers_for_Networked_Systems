import os
import shutil
import tempfile

import numpy as np

from envs.coupled_msd_env import CoupledMSDEnv, DEFAULT_D, DEFAULT_M, SINE_K, complete_graph


def check_environment():
    from gymnasium.utils.env_checker import check_env

    check_env(CoupledMSDEnv("sine"))
    print("env_checker: ok")


def check_deterministic_rollout():
    env1 = CoupledMSDEnv("sine")
    env2 = CoupledMSDEnv("sine")
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    assert np.allclose(obs1, obs2)
    action = np.zeros(env1.action_space.shape, dtype=np.float32)
    for _ in range(10):
        obs1, r1, term1, trunc1, _ = env1.step(action)
        obs2, r2, term2, trunc2, _ = env2.step(action)
        assert np.allclose(obs1, obs2)
        assert np.isclose(r1, r2)
        assert term1 == term2 and trunc1 == trunc2
    print("deterministic_rollout: ok")


def old_sine_euler_step(state, normalized_action):
    n = 5
    dt = 0.02
    u = 12.0 * np.asarray(normalized_action, dtype=np.float32)
    q = state[:n]
    v = state[n:]
    A = complete_graph(n, coupling=0.5)
    degree = np.sum(A, axis=1)
    z = A @ q - degree * q
    q_next = q + dt * v
    v_next = v + dt * (-DEFAULT_D * v + SINE_K * np.sin(q) + z + u) / DEFAULT_M
    return np.concatenate([q_next, v_next]).astype(np.float32)


def check_old_new_dynamics():
    env = CoupledMSDEnv("sine", integrator="euler")
    state = np.array([0.2, -0.3, 0.1, -0.4, 0.5, 0.05, -0.02, 0.03, -0.04, 0.01], dtype=np.float32)
    action = np.array([0.2, -0.1, 0.0, 0.4, -0.3], dtype=np.float32)
    env.reset(options={"state": state})
    obs, _, _, _, _ = env.step(action)
    ref = old_sine_euler_step(state, action)
    assert np.allclose(obs, ref, atol=1e-7)
    print("old_vs_new_euler_dynamics: ok")


def check_td3_smoke():
    from tianshou.data import Collector, ReplayBuffer
    from tianshou.utils.torch_utils import policy_within_training_step

    from residual_policy import build_td3, load_model, save_model

    env = CoupledMSDEnv("sine")
    config = {
        "n_subsystems": env.n,
        "adjacency": env.A.tolist(),
        "linear_init": [-4.8, -1.2, 0.0],
        "action_scale": env.umax,
        "umax": env.umax,
        "actor_mode": "residual",
        "actor_hidden_layers": [16],
        "critic_hidden_layers": [16],
        "actor_activation": "elu",
        "critic_activation": "elu",
        "actor_lr": 5e-6,
        "critic_lr": 1e-4,
        "gamma": 0.98,
        "tau": 0.005,
        "exploration_noise": 0.01,
        "policy_noise": 0.01,
        "noise_clip": 0.5,
        "policy_delay": 2,
    }
    algorithm = build_td3(env, config, "cpu")
    buffer = ReplayBuffer(1000)
    collector = Collector(algorithm, env, buffer, exploration_noise=True)
    collector.reset()
    for _ in range(32):
        with policy_within_training_step(algorithm.policy):
            collector.collect(n_step=1)
            if len(buffer) >= 16:
                algorithm.update(buffer=buffer, sample_size=16)

    tmp = tempfile.mkdtemp(prefix="td3_smoke_")
    try:
        path = os.path.join(tmp, "model.pth")
        save_model(path, algorithm, config)
        loaded = load_model(path, env, device="cpu")
        obs, _ = env.reset(seed=321)
        action, _ = loaded.predict(obs, deterministic=True)
        assert action.shape == env.action_space.shape
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("td3_smoke_training: ok")


def main():
    check_environment()
    check_deterministic_rollout()
    check_old_new_dynamics()
    check_td3_smoke()


if __name__ == "__main__":
    main()
