"""Overcooked environment wrapper.

Cooperative cooking game: agents pick up ingredients, cook, plate, and deliver.
Only 2 agents, so 1 Byzantine = fully adversarial (edge case for BAHT).

Install: pip install overcooked-ai
"""

import numpy as np
import gym
from gym import spaces

try:
    # Fix numpy 2.0 compatibility before importing overcooked
    import numpy
    if not hasattr(numpy, "Inf"):
        numpy.Inf = numpy.inf
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    OVERCOOKED_AVAILABLE = True
except ImportError:
    OVERCOOKED_AVAILABLE = False


class OvercookedGymEnv(gym.Env):
    """Gym wrapper for Overcooked compatible with GymmaWrapper."""

    NUM_ACTIONS = 6
    LAYOUTS = [
        "cramped_room",
        "asymmetric_advantages",
        "coordination_ring",
        "forced_coordination",
        "counter_circuit",
    ]

    def __init__(
        self,
        layout_name: str = "cramped_room",
        horizon: int = 400,
        reward_shaping: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not OVERCOOKED_AVAILABLE:
            raise ImportError("overcooked-ai not installed. Run: pip install overcooked-ai")

        self.layout_name = layout_name
        self.horizon = horizon
        self.reward_shaping = reward_shaping
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.n_agents = 2
        self.episode_limit = horizon

        n_pots = len(self.mdp.get_pot_locations())
        self.obs_size = self.n_agents * 11 + n_pots * 5

        self.action_space = spaces.Tuple([spaces.Discrete(self.NUM_ACTIONS) for _ in range(self.n_agents)])
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

        self.state = None
        self.t = 0

    def _get_featurized_obs(self, agent_id):
        if self.state is None:
            return np.zeros(self.obs_size, dtype=np.float32)

        features = []
        height, width = self.mdp.shape

        for player in self.state.players:
            pos = player.position
            features.extend([pos[0] / width, pos[1] / height])

            orientation = player.orientation
            ori_onehot = [0, 0, 0, 0]
            if orientation == (0, -1):
                ori_onehot[0] = 1
            elif orientation == (0, 1):
                ori_onehot[1] = 1
            elif orientation == (1, 0):
                ori_onehot[2] = 1
            elif orientation == (-1, 0):
                ori_onehot[3] = 1
            features.extend(ori_onehot)

            held = player.held_object
            held_onehot = [0, 0, 0, 0, 0]
            if held is None:
                held_onehot[0] = 1
            elif held.name == "onion":
                held_onehot[1] = 1
            elif held.name == "tomato":
                held_onehot[2] = 1
            elif held.name == "dish":
                held_onehot[3] = 1
            elif "soup" in held.name:
                held_onehot[4] = 1
            features.extend(held_onehot)

        for pot_pos in self.mdp.get_pot_locations():
            features.extend([pot_pos[0] / width, pot_pos[1] / height])
            if self.state.has_object(pot_pos):
                pot_state = self.state.get_object(pot_pos)
                n_items = len(pot_state.ingredients) if hasattr(pot_state, "ingredients") else 0
                cooking = pot_state.cooking_tick if hasattr(pot_state, "cooking_tick") else 0
                is_ready = 1 if (hasattr(pot_state, "is_ready") and pot_state.is_ready) else 0
                features.extend([n_items / 3, cooking / 20, is_ready])
            else:
                features.extend([0, 0, 0])

        features = np.array(features, dtype=np.float32)
        if len(features) < self.obs_size:
            features = np.pad(features, (0, self.obs_size - len(features)))
        elif len(features) > self.obs_size:
            features = features[:self.obs_size]
        return features

    def _action_to_overcooked(self, action):
        action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0), 5: "interact"}
        return action_map.get(action, (0, 0))

    def step(self, actions):
        joint_action = tuple(self._action_to_overcooked(a) for a in actions)
        next_state, info_dict = self.mdp.get_state_transition(self.state, joint_action)
        self.state = next_state
        self.t += 1

        if self.reward_shaping:
            reward = sum(info_dict.get("shaped_reward_by_agent", [0, 0]))
        else:
            reward = sum(info_dict.get("sparse_reward_by_agent", [0, 0]))

        done = self.t >= self.horizon
        obs = tuple(self._get_featurized_obs(i) for i in range(self.n_agents))

        return obs, [float(reward)] * self.n_agents, [done] * self.n_agents, {"episode_limit": done}

    def reset(self):
        self.state = self.mdp.get_standard_start_state()
        self.t = 0
        return tuple(self._get_featurized_obs(i) for i in range(self.n_agents))

    def render(self, mode="human"):
        if self.state is not None:
            print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
