import numpy as np
import gymnasium as gym
from gymnasium import spaces


DEFAULT_M = np.array([1.00, 1.20, 0.90, 1.10, 1.00], dtype=np.float32)
DEFAULT_D = np.array([0.10, 0.08, 0.12, 0.07, 0.11], dtype=np.float32)
MSD_K = np.array([0.20, 0.25, 0.15, 0.30, 0.18], dtype=np.float32)
SINE_K = np.array([4.00, 3.50, 4.00 + 0.50, 3.00, 4.00], dtype=np.float32)


def component_param(value, default, n):
    if isinstance(value, str):
        if value.lower() == "default":
            return np.resize(default, n).astype(np.float32)
        value = [float(item.strip()) for item in value.split(",")]

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (n,):
        raise ValueError(f"expected {n} values, got shape {array.shape}")
    return array


def make_graph(n, graph, coupling=0.5):
    adjacency = np.zeros((n, n), dtype=np.float32)
    if graph == "none":
        return adjacency
    if graph == "complete":
        return coupling * (np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32))
    if graph in {"chain", "ring"}:
        for i in range(n - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = coupling
        if graph == "ring" and n > 2:
            adjacency[0, -1] = adjacency[-1, 0] = coupling
        return adjacency
    if graph == "star":
        adjacency[0, 1:] = coupling
        adjacency[1:, 0] = coupling
        return adjacency
    raise ValueError(f"unknown graph: {graph}")


def complete_graph(n=5, coupling=0.5):
    return make_graph(n, "complete", coupling)


class CoupledMSDEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_type="sine",
        n=5,
        graph="complete",
        coupling=0.5,
        m="default",
        d="default",
        k="default",
        dt=None,
        episode_len=None,
        umax=None,
        q_weight=1.0,
        v_weight=0.3,
        u_weight=None,
        terminal_weight=0.0,
        q_init_low=None,
        q_init_high=None,
        v_init_low=None,
        v_init_high=None,
        integrator="rk4",
    ):
        super().__init__()
        if env_type not in {"msd", "sine"}:
            raise ValueError(f"unknown env_type: {env_type}")
        if integrator not in {"euler", "rk4"}:
            raise ValueError(f"unknown integrator: {integrator}")

        self.env_type = env_type
        self.integrator = integrator
        self.n = n
        self.dt = dt if dt is not None else (0.02 if env_type == "sine" else 0.05)
        self.episode_len = episode_len if episode_len is not None else (250 if env_type == "sine" else 200)
        self.umax = umax if umax is not None else (12.0 if env_type == "sine" else 1.0)
        self.q_weight = q_weight
        self.v_weight = v_weight
        self.u_weight = u_weight if u_weight is not None else (0.03 if env_type == "sine" else 0.05)
        self.terminal_weight = terminal_weight
        self.q_init_low = q_init_low if q_init_low is not None else (-2.4 if env_type == "sine" else -0.5)
        self.q_init_high = q_init_high if q_init_high is not None else (2.4 if env_type == "sine" else 0.5)
        self.v_init_low = v_init_low if v_init_low is not None else (-0.4 if env_type == "sine" else -0.2)
        self.v_init_high = v_init_high if v_init_high is not None else (0.4 if env_type == "sine" else 0.2)

        self.m = component_param(m, DEFAULT_M, self.n)
        self.d = component_param(d, DEFAULT_D, self.n)
        default_k = SINE_K if env_type == "sine" else MSD_K
        self.k = component_param(k, default_k, self.n)
        self.A = make_graph(self.n, graph, coupling)
        self.degree = np.sum(self.A, axis=1)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.n,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n,),
            dtype=np.float32,
        )

        self.state = None
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float32)
        else:
            q0 = self.np_random.uniform(self.q_init_low, self.q_init_high, self.n)
            v0 = self.np_random.uniform(self.v_init_low, self.v_init_high, self.n)
            state = np.concatenate([q0, v0]).astype(np.float32)
        self.state = state
        self.t = 0
        return self.state.copy(), self._info(np.zeros(self.n, dtype=np.float32))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        u = self.umax * action
        next_state = self._dynamics(self.state, u)
        self.t += 1
        truncated = self.t >= self.episode_len
        reward, costs = self._reward(next_state, u, truncated)
        self.state = next_state
        terminated = bool(not np.all(np.isfinite(self.state)))
        info = self._info(u)
        info.update(costs)
        return self.state.copy(), reward, terminated, truncated, info

    def _coupling_force(self, q):
        return self.A @ q - self.degree * q

    def _rhs(self, state, u):
        q = state[:self.n]
        v = state[self.n:]
        z = self._coupling_force(q)
        if self.env_type == "sine":
            force = -self.d * v + self.k * np.sin(q) + z + u
        else:
            force = -self.d * v - self.k * q + z + u
        return np.concatenate([v, force / self.m]).astype(np.float32)

    def _dynamics(self, state, u):
        state = np.asarray(state, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        if self.integrator == "euler":
            return (state + self.dt * self._rhs(state, u)).astype(np.float32)

        k1 = self._rhs(state, u)
        k2 = self._rhs(state + 0.5 * self.dt * k1, u)
        k3 = self._rhs(state + 0.5 * self.dt * k2, u)
        k4 = self._rhs(state + self.dt * k3, u)
        return (state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)

    def _reward(self, next_state, u, truncated):
        q = next_state[:self.n]
        v = next_state[self.n:]
        cost_q = self.q_weight * float(np.sum(q ** 2))
        cost_v = self.v_weight * float(np.sum(v ** 2))
        cost_u = self.u_weight * float(np.sum(u ** 2))
        cost_terminal = self.terminal_weight * float(np.sum(next_state ** 2)) if truncated else 0.0
        return -(cost_q + cost_v + cost_u + cost_terminal), {
            "cost_q": cost_q,
            "cost_v": cost_v,
            "cost_u": cost_u,
            "cost_terminal": cost_terminal,
        }

    def _info(self, u):
        q = self.state[:self.n]
        v = self.state[self.n:]
        z = self._coupling_force(q)
        return {
            "q": q.copy(),
            "v": v.copy(),
            "z": z.copy(),
            "u": np.asarray(u, dtype=np.float32).copy(),
            "state_norm": float(np.linalg.norm(self.state)),
            "action_norm": float(np.linalg.norm(u)),
        }
