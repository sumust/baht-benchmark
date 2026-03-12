"""Environment wrappers for the BAHT benchmark suite.

All environments implement the MultiAgentEnv interface:
  - reset() -> (obs_list, state)
  - step(actions) -> (reward, done, info)
  - get_obs() -> list of obs arrays
  - get_state() -> state array
  - get_avail_actions() -> list of avail action arrays
  - get_env_info() -> dict with state_shape, obs_shape, n_actions, n_agents, episode_limit
"""
