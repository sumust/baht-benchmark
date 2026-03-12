"""SMAC (StarCraft Multi-Agent Challenge) wrapper.

Large-team micromanagement scenarios. Many agents = more Byzantines possible.
Requires StarCraft II installation + pysc2.

Install: pip install pysc2 s2clientprotocol; bash install_sc2.sh
"""

# SMAC provides its own MultiAgentEnv interface natively.
# No wrapper needed — just import and use directly.
# The env_key is "sc2" and it's handled by the NAHT/EPyMARL env registry.

try:
    from smac.env import StarCraft2Env  # noqa: F401
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False


# Available SMAC maps suitable for BAHT
SMAC_MAPS = {
    "3m": {"n_agents": 3, "n_actions": 9, "episode_limit": 60},
    "3s5z": {"n_agents": 8, "n_actions": 14, "episode_limit": 150},
    "5m_vs_6m": {"n_agents": 5, "n_actions": 12, "episode_limit": 70},
    "8m": {"n_agents": 8, "n_actions": 14, "episode_limit": 120},
    "10m_vs_11m": {"n_agents": 10, "n_actions": 17, "episode_limit": 150},
    "3s_vs_5z": {"n_agents": 3, "n_actions": 11, "episode_limit": 150},
}
