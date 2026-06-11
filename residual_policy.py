import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tianshou.algorithm import TD3
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactory
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import ModuleWithVectorOutput, Net
from tianshou.utils.net.continuous import (
    AbstractContinuousActorDeterministic,
    ContinuousCritic,
)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


class ExponentialLRSchedulerFactory(LRSchedulerFactory):
    def __init__(self, initial_lr, decay_rate, decay_steps, min_lr):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr

    def create_scheduler(self, optim):
        min_factor = self.min_lr / self.initial_lr
        return LambdaLR(
            optim,
            lr_lambda=lambda update: max(
                min_factor,
                self.decay_rate ** (update / self.decay_steps),
            ),
        )


def optimizer_factory(config, prefix):
    initial_lr = config[f"{prefix}_lr"]
    scheduler = ExponentialLRSchedulerFactory(
        initial_lr=initial_lr,
        decay_rate=config.get(f"{prefix}_lr_decay_rate", 1.0),
        decay_steps=config.get("lr_decay_steps", 10_000),
        min_lr=config.get(f"min_{prefix}_lr", 0.0),
    )
    return AdamOptimizerFactory(lr=initial_lr).with_lr_scheduler_factory(
        scheduler
    )


class ResidualController(nn.Module):
    def __init__(
        self,
        n_subsystems,
        adjacency,
        hidden_layers,
        activation_fn,
        linear_init,
        action_scale,
        umax,
        actor_mode,
    ):
        super().__init__()
        self.n = n_subsystems
        self.action_scale = action_scale
        self.umax = umax
        self.actor_mode = actor_mode

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("degree", adjacency.sum(dim=1))
        linear_init = torch.as_tensor(linear_init, dtype=torch.float32)
        if linear_init.ndim == 1:
            linear_init = linear_init.repeat(self.n, 1)
        if linear_init.shape != (self.n, 3):
            raise ValueError("linear_init must have shape (3,) or (N, 3).")
        self.linear_gain = nn.Parameter(linear_init)
        self.local_nets = nn.ModuleList()
        if self.actor_mode == "residual":
            self.local_nets.extend(
                self._make_local_net(hidden_layers, activation_fn)
                for _ in range(self.n)
            )

    @staticmethod
    def _make_local_net(hidden_layers, activation_fn):
        layers = []
        input_dim = 3
        for width in hidden_layers:
            layers.extend([nn.Linear(input_dim, width), activation_fn()])
            input_dim = width
        output = nn.Linear(input_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)

    def forward(self, state):
        q = state[:, : self.n]
        v = state[:, self.n :]
        z = q @ self.adjacency.T - q * self.degree
        features = torch.stack((q, v, z), dim=-1)

        linear = torch.sum(self.linear_gain.unsqueeze(0) * features, dim=-1)
        if self.actor_mode == "residual":
            residual = torch.cat(
                [net(features[:, i, :]) for i, net in enumerate(self.local_nets)],
                dim=1,
            )
        else:
            residual = torch.zeros_like(linear)
        physical_action = self.action_scale * torch.tanh(
            (linear + residual) / self.action_scale
        )
        return physical_action / self.umax


class ResidualActor(AbstractContinuousActorDeterministic):
    def __init__(self, config):
        super().__init__(config["n_subsystems"])
        self.max_action = 1.0
        self.controller = ResidualController(
            n_subsystems=config["n_subsystems"],
            adjacency=config["adjacency"],
            hidden_layers=config["actor_hidden_layers"],
            activation_fn=ACTIVATIONS[config["actor_activation"]],
            linear_init=config["linear_init"],
            action_scale=config["action_scale"],
            umax=config["umax"],
            actor_mode=config["actor_mode"],
        )

    def get_preprocess_net(self):
        return ModuleWithVectorOutput.from_module(
            nn.Identity(), 2 * self.controller.n
        )

    def forward(self, obs, state=None, info=None):
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return self.controller(obs), state


class TD3Model:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def predict(self, observation, deterministic=True):
        self.algorithm.eval()
        with torch.no_grad():
            action = self.algorithm.policy.compute_action(observation)
        return np.asarray(action, dtype=np.float32), None


def build_td3(env, config, device):
    actor = ResidualActor(config).to(device)
    critic_net1 = Net(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_sizes=config["critic_hidden_layers"],
        activation=ACTIVATIONS[config["critic_activation"]],
        concat=True,
    )
    critic_net2 = Net(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_sizes=config["critic_hidden_layers"],
        activation=ACTIVATIONS[config["critic_activation"]],
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=critic_net1).to(device)
    critic2 = ContinuousCritic(preprocess_net=critic_net2).to(device)
    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(
            sigma=config["exploration_noise"] / config["umax"]
        ),
        action_space=env.action_space,
        observation_space=env.observation_space,
        action_scaling=True,
        action_bound_method="clip",
    )
    algorithm = TD3(
        policy=policy,
        policy_optim=optimizer_factory(config, "actor"),
        critic=critic1,
        critic_optim=optimizer_factory(config, "critic"),
        critic2=critic2,
        critic2_optim=optimizer_factory(config, "critic"),
        tau=config["tau"],
        gamma=config["gamma"],
        policy_noise=config["policy_noise"] / config["umax"],
        update_actor_freq=config["policy_delay"],
        noise_clip=config["noise_clip"] / config["umax"],
    )
    return algorithm


def save_model(path, algorithm, config):
    torch.save(
        {"state_dict": algorithm.state_dict(), "config": config},
        path,
    )


def load_model(path, env, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    algorithm = build_td3(env, checkpoint["config"], device)
    algorithm.load_state_dict(checkpoint["state_dict"])
    algorithm.eval()
    return TD3Model(algorithm)
