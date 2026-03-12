"""Advisor Trust Environment — a BAHT benchmark where detection directly impacts performance.

Setup: 1 ego agent + N-1 advisor agents. Each step has a hidden "correct" action.
Honest advisors recommend correctly, Byzantine advisors lie.
With detection, ego can ignore Byzantine advisors and follow honest ones.

Built-in — no external dependencies.
"""

import numpy as np
from baht_benchmark.envs.base import MultiAgentEnv


class AdvisorTrustEnv:
    """Core Advisor Trust environment."""

    def __init__(
        self,
        n_advisors: int = 4,
        byzantine_ratio: float = 0.5,
        adversarial_mode: str = "opposite",
        episode_length: int = 20,
        seed: int = None,
    ):
        self.n_advisors = n_advisors
        self.n_agents = n_advisors + 1  # advisors + ego
        self.byzantine_ratio = byzantine_ratio
        self.adversarial_mode = adversarial_mode
        self.episode_length = episode_length

        self.n_actions = 2
        self.obs_dim = self.n_advisors * self.n_actions + self.n_actions

        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.is_byzantine = np.zeros(n_advisors, dtype=bool)
        self.byzantine_labels = np.zeros(self.n_agents, dtype=np.float32)
        self.correct_action = 0
        self.advisor_recommendations = np.zeros(n_advisors, dtype=np.int32)
        self.last_actions = np.zeros(self.n_agents, dtype=np.int32)

    def reset(self):
        self.t = 0
        n_byzantine = int(self.n_advisors * self.byzantine_ratio)
        byzantine_indices = self.rng.choice(self.n_advisors, size=n_byzantine, replace=False)
        self.is_byzantine = np.zeros(self.n_advisors, dtype=bool)
        self.is_byzantine[byzantine_indices] = True

        self.byzantine_labels = np.zeros(self.n_agents, dtype=np.float32)
        self.byzantine_labels[1:] = self.is_byzantine.astype(np.float32)

        self.correct_action = self.rng.randint(0, self.n_actions)
        self.advisor_recommendations = self._generate_recommendations()
        self.last_actions = np.zeros(self.n_agents, dtype=np.int32)

        return self._get_obs()

    def _generate_recommendations(self):
        recommendations = np.zeros(self.n_advisors, dtype=np.int32)
        for i in range(self.n_advisors):
            if self.is_byzantine[i]:
                if self.adversarial_mode == "opposite":
                    recommendations[i] = 1 - self.correct_action
                elif self.adversarial_mode == "random":
                    recommendations[i] = self.rng.randint(0, self.n_actions)
                elif self.adversarial_mode == "strategic":
                    if self.rng.random() < 0.8:
                        recommendations[i] = 1 - self.correct_action
                    else:
                        recommendations[i] = self.correct_action
            else:
                recommendations[i] = self.correct_action
        return recommendations

    def _get_obs(self):
        obs = []
        ego_obs = np.zeros(self.obs_dim, dtype=np.float32)
        for i, rec in enumerate(self.advisor_recommendations):
            ego_obs[i * self.n_actions + rec] = 1.0
        ego_obs[self.n_advisors * self.n_actions + self.last_actions[0]] = 1.0
        obs.append(ego_obs)

        for i in range(self.n_advisors):
            obs.append(ego_obs.copy())
        return obs

    def step(self, actions):
        ego_action = actions[0]
        reward = 1.0 if ego_action == self.correct_action else -1.0
        self.last_actions = np.array(actions, dtype=np.int32)

        self.t += 1
        done = self.t >= self.episode_length

        if not done:
            self.correct_action = self.rng.randint(0, self.n_actions)
            self.advisor_recommendations = self._generate_recommendations()

        info = {
            "correct_action": self.correct_action,
            "ego_action": ego_action,
            "byzantine_labels": self.byzantine_labels.copy(),
        }
        return self._get_obs(), [reward] * self.n_agents, [done] * self.n_agents, info

    def get_state(self):
        state = np.concatenate([
            self._get_obs()[0],
            self.byzantine_labels,
            [self.correct_action],
        ])
        return state.astype(np.float32)

    def get_avail_actions(self):
        return [np.ones(self.n_actions, dtype=np.int32) for _ in range(self.n_agents)]

    def get_byzantine_labels(self):
        return self.byzantine_labels.copy()


class AdvisorTrustMARL(MultiAgentEnv):
    """MARL-compatible wrapper matching EPyMARL/NAHT MultiAgentEnv interface."""

    def __init__(self, n_advisors=4, byzantine_ratio=0.5, adversarial_mode="opposite",
                 episode_length=20, seed=None, **kwargs):
        self.env = AdvisorTrustEnv(
            n_advisors=n_advisors,
            byzantine_ratio=byzantine_ratio,
            adversarial_mode=adversarial_mode,
            episode_length=episode_length,
            seed=seed,
        )
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions
        self.episode_limit = self.env.episode_length
        self._obs = None
        self._state = None

    def reset(self):
        self._obs = self.env.reset()
        self._state = self.env.get_state()
        return self._obs, self._state

    def step(self, actions):
        self._obs, rewards, dones, info = self.env.step(actions)
        self._state = self.env.get_state()
        return rewards[0], dones[0], info

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self.env.obs_dim

    def get_state(self):
        return self._state

    def get_state_size(self):
        return len(self.env.get_state())

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return self.n_actions

    def get_byzantine_labels(self):
        return self.env.get_byzantine_labels()

    def close(self):
        pass

    def render(self):
        pass

    def save_replay(self):
        pass
