"""Level-Based Foraging (LBF) environment wrapper.

Cooperative multi-agent foraging: agents coordinate to pick up food.
Food can only be collected if combined levels of adjacent agents >= food level.
Ideal for BAHT — Byzantine can refuse to help, blocking collection.

Install: pip install lbforaging
"""

import numpy as np
import gym
from gym import spaces
from baht_benchmark.envs.base import MultiAgentEnv

try:
    import lbforaging
    LBF_AVAILABLE = True
except ImportError:
    LBF_AVAILABLE = False


class LBFGymEnv(gym.Env):
    """Gym wrapper for LBF compatible with GymmaWrapper."""

    NUM_ACTIONS = 6

    def __init__(
        self,
        field_size: int = 8,
        n_players: int = 3,
        max_food: int = 2,
        force_coop: bool = True,
        max_episode_steps: int = 50,
        sight: int = 8,
        normalize_obs: bool = True,
        reward_shaping: bool = True,
        **kwargs,
    ):
        super().__init__()

        if not LBF_AVAILABLE:
            raise ImportError("lbforaging not installed. Run: pip install lbforaging")

        self.field_size = field_size
        self.n_players = n_players
        self.n_agents = n_players
        self.max_food = max_food
        self.force_coop = force_coop
        self.max_episode_steps = max_episode_steps
        self.episode_limit = max_episode_steps
        self.sight = min(sight, field_size)
        self.normalize_obs = normalize_obs
        self.reward_shaping = reward_shaping

        coop_str = "-coop" if force_coop else ""
        env_name = f"Foraging-{field_size}x{field_size}-{n_players}p-{max_food}f{coop_str}-v3"

        try:
            import gymnasium
            self._env = gymnasium.make(env_name, max_episode_steps=max_episode_steps, sight=self.sight)
            self._use_gymnasium = True
        except Exception:
            self._env = gym.make(env_name)
            self._use_gymnasium = False

        if self._use_gymnasium:
            sample_obs, _ = self._env.reset()
        else:
            sample_obs = self._env.reset()

        self.obs_size = sample_obs[0].shape[0]

        self.action_space = spaces.Tuple([spaces.Discrete(self.NUM_ACTIONS) for _ in range(self.n_agents)])
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

        self.t = 0
        self._total_reward = 0

    def _process_obs(self, obs):
        processed = []
        for o in obs:
            o = np.array(o, dtype=np.float32).flatten()
            if self.normalize_obs:
                o = np.clip(o / (self.field_size + 1), -1, 1)
            processed.append(o)
        return tuple(processed)

    def step(self, actions):
        actions_array = np.array(actions, dtype=np.int64)

        if self._use_gymnasium:
            obs, rewards, terminated, truncated, info_dict = self._env.step(actions_array)
            done = terminated or truncated
        else:
            obs, rewards, dones_raw, info_dict = self._env.step(actions_array)
            done = all(dones_raw) if hasattr(dones_raw, "__iter__") else dones_raw
            info_dict = info_dict if isinstance(info_dict, dict) else {}

        self.t += 1
        processed_obs = self._process_obs(obs)

        if isinstance(rewards, (list, tuple, np.ndarray)):
            base_reward = float(sum(rewards))
        else:
            base_reward = float(rewards)

        shaped_reward = base_reward * 10.0 if (self.reward_shaping and base_reward > 0) else base_reward
        self._total_reward += shaped_reward

        rewards_list = [shaped_reward] * self.n_agents
        dones = [done] * self.n_agents
        info = {"episode_limit": done, "t": self.t, "base_reward": base_reward}

        return processed_obs, rewards_list, dones, info

    def reset(self):
        if self._use_gymnasium:
            obs, _ = self._env.reset()
        else:
            obs = self._env.reset()
        self.t = 0
        self._total_reward = 0
        return self._process_obs(obs)

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self._env, "seed"):
                return self._env.seed(seed)
        return [seed]
