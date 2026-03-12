"""DSSE (Drone Swarm Search Environment) wrapper.

4 drones search for shipwrecked targets using a shared probability matrix.
Physical coordination required — ideal for BAHT (Byzantine can waste coverage area).

Install: pip install DSSE
"""

import numpy as np
from baht_benchmark.envs.base import MultiAgentEnv

try:
    from DSSE import DroneSwarmSearch
    DSSE_AVAILABLE = True
except ImportError:
    DSSE_AVAILABLE = False


class DSSEWrapper(MultiAgentEnv):
    """MultiAgentEnv wrapper for DSSE (Drone Swarm Search Environment)."""

    NUM_ACTIONS = 9

    def __init__(
        self,
        grid_size: int = 15,
        n_drones: int = 4,
        n_targets: int = 2,
        timestep_limit: int = 100,
        drone_speed: int = 1,
        probability_of_detection: float = 0.9,
        normalize_obs: bool = True,
        include_other_positions: bool = True,
        seed: int = None,
        **kwargs,
    ):
        if not DSSE_AVAILABLE:
            raise ImportError("DSSE not installed. Run: pip install DSSE")

        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_agents = n_drones
        self.n_targets = n_targets
        self.timestep_limit = timestep_limit
        self.episode_limit = timestep_limit
        self.normalize_obs = normalize_obs
        self.include_other_positions = include_other_positions
        self._seed = seed

        if seed is not None:
            np.random.seed(seed)

        self._env = DroneSwarmSearch(
            grid_size=grid_size,
            render_mode="ansi",
            render_grid=False,
            render_gradient=False,
            vector=(0.5, 0.5),
            timestep_limit=timestep_limit,
            person_amount=n_targets,
            dispersion_inc=0.03,
            drone_amount=n_drones,
            drone_speed=drone_speed,
            probability_of_detection=probability_of_detection,
            pre_render_time=0,
        )

        self._position_dim = 2
        self._prob_matrix_dim = grid_size * grid_size
        self._other_pos_dim = (n_drones - 1) * 2 if include_other_positions else 0
        self._obs_size = self._position_dim + self._prob_matrix_dim + self._other_pos_dim

        self.t = 0
        self._agent_names = None
        self._obs = None
        self._drone_positions = {}

    def reset(self):
        obs_dict, info = self._env.reset()
        self.t = 0
        self._agent_names = list(self._env.get_agents())
        self._obs = self._process_observations(obs_dict)
        return self._obs, self.get_state()

    def step(self, actions):
        agent_names = self._agent_names
        action_dict = {}
        for i, name in enumerate(agent_names):
            if name in self._env.agents:
                action_dict[name] = int(actions[i])

        obs_dict, rewards_dict, terms_dict, truncs_dict, info_dict = self._env.step(action_dict)
        self.t += 1
        self._obs = self._process_observations(obs_dict)

        total_reward = sum(rewards_dict.values()) if rewards_dict else 0.0
        terminated = any(terms_dict.values()) or any(truncs_dict.values()) if terms_dict else False
        terminated = terminated or (self.t >= self.timestep_limit)

        info = {"episode_limit": self.t >= self.timestep_limit}
        return total_reward, terminated, info

    def _process_observations(self, obs_dict):
        agent_names = self._agent_names
        processed = []

        positions = {}
        for name in agent_names:
            if name in obs_dict:
                pos, _ = obs_dict[name]
                positions[name] = pos
                self._drone_positions[name] = pos

        for i, name in enumerate(agent_names):
            if name not in obs_dict:
                processed.append(np.zeros(self._obs_size, dtype=np.float32))
                continue

            pos, prob_matrix = obs_dict[name]
            pos_flat = np.array([pos[0], pos[1]], dtype=np.float32)
            if self.normalize_obs:
                pos_flat = pos_flat / self.grid_size

            prob_flat = np.array(prob_matrix, dtype=np.float32).flatten()
            if self.normalize_obs:
                prob_flat = np.clip(prob_flat, 0, 1)

            obs_parts = [pos_flat, prob_flat]

            if self.include_other_positions:
                other_pos = []
                for j, other_name in enumerate(agent_names):
                    if j != i:
                        other_p = positions.get(other_name, (0, 0))
                        if self.normalize_obs:
                            other_pos.extend([other_p[0] / self.grid_size, other_p[1] / self.grid_size])
                        else:
                            other_pos.extend([other_p[0], other_p[1]])
                obs_parts.append(np.array(other_pos, dtype=np.float32))

            obs = np.concatenate(obs_parts)
            processed.append(obs)

        return processed

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.NUM_ACTIONS, dtype=np.int32)

    def get_total_actions(self):
        return self.NUM_ACTIONS

    def render(self, mode="human"):
        pass

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def save_replay(self):
        pass
