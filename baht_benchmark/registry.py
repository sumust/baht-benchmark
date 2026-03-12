"""Environment registry for the BAHT benchmark suite."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvConfig:
    """Configuration for a benchmark environment."""
    name: str
    env_key: str  # Key used to instantiate the env
    n_agents: int
    n_actions: int
    episode_limit: int
    obs_shape: int  # Approximate; actual depends on config
    interface: str  # "native" (MultiAgentEnv) or "gymma" (via GymmaWrapper)
    install: str  # pip install command or "built-in"
    tier: str  # "core" or "extended"
    description: str

    # BAHT-specific
    min_agents_for_baht: int = 3  # Minimum agents for meaningful BAHT
    recommended_byzantine_budget: int = 1

    # Training defaults
    default_t_max: int = 250_000
    default_batch_size: int = 32
    default_buffer_size: int = 32
    default_batch_size_run: int = 4
    default_hidden_dim: int = 64

    # Domain randomization noise calibration
    obs_noise_std: float = 0.02
    action_noise_eps: float = 0.05

    # Environment-specific kwargs
    env_kwargs: Dict = field(default_factory=dict)

    # Pretrain config
    pretrain_t_max: int = 100_000
    pretrain_seeds: int = 3


# ─── Core Environments ───────────────────────────────────────────────

MPE_PP = EnvConfig(
    name="mpe-pp",
    env_key="mpe",
    n_agents=4,
    n_actions=5,
    episode_limit=25,
    obs_shape=18,
    interface="native",
    install="./3rdparty/mpe",
    tier="core",
    description="MPE Predator-Prey: 3 predators chase 1 prey. Physical coordination required.",
    recommended_byzantine_budget=1,
    default_t_max=250_000,
    default_batch_size=32,
    default_buffer_size=32,
    default_batch_size_run=4,
    default_hidden_dim=64,
    obs_noise_std=0.05,
    action_noise_eps=0.10,
    env_kwargs={
        "key": "mpe:SimplePredPreyHeuristic-v0",
        "time_limit": 25,
    },
)

DSSE = EnvConfig(
    name="dsse",
    env_key="dsse",
    n_agents=4,
    n_actions=9,
    episode_limit=100,
    obs_shape=233,  # 2 + 15*15 + 3*2 = 233
    interface="native",
    install="pip install DSSE",
    tier="core",
    description="Drone Swarm Search: 4 drones search for shipwrecked targets. Coverage coordination.",
    recommended_byzantine_budget=1,
    default_t_max=500_000,
    default_batch_size=32,
    default_buffer_size=32,
    default_batch_size_run=4,
    default_hidden_dim=64,
    obs_noise_std=0.02,
    action_noise_eps=0.05,
    env_kwargs={
        "grid_size": 15,
        "n_drones": 4,
        "n_targets": 2,
        "timestep_limit": 100,
    },
)

LBF = EnvConfig(
    name="lbf",
    env_key="gymma",
    n_agents=3,
    n_actions=6,
    episode_limit=50,
    obs_shape=27,
    interface="gymma",
    install="pip install lbforaging",
    tier="core",
    description="Level-Based Foraging: agents must coordinate to collect food. Cooperation required.",
    min_agents_for_baht=3,
    recommended_byzantine_budget=1,
    default_t_max=250_000,
    default_batch_size=32,
    default_buffer_size=32,
    default_batch_size_run=4,
    default_hidden_dim=64,
    obs_noise_std=0.03,
    action_noise_eps=0.08,
    env_kwargs={
        "key": "lbforaging:Foraging-8x8-3p-2f-coop-v3",
        "time_limit": 50,
    },
)

ADVISOR_TRUST = EnvConfig(
    name="advisor-trust",
    env_key="advisor_trust",
    n_agents=5,
    n_actions=2,
    episode_limit=20,
    obs_shape=10,  # 4*2 + 2 = 10
    interface="native",
    install="built-in",
    tier="core",
    description="Advisor Trust: ego follows/ignores advisor recommendations. Detection = weighted voting.",
    min_agents_for_baht=3,
    recommended_byzantine_budget=2,
    default_t_max=50_000,
    default_batch_size=64,
    default_buffer_size=256,
    default_batch_size_run=8,
    default_hidden_dim=16,
    obs_noise_std=0.01,
    action_noise_eps=0.05,
    env_kwargs={
        "n_advisors": 4,
        "byzantine_ratio": 0.5,
        "adversarial_mode": "opposite",
        "episode_length": 20,
    },
)

MATRIX_GAMES = EnvConfig(
    name="matrix-games",
    env_key="gymma",
    n_agents=2,
    n_actions=2,
    episode_limit=1,
    obs_shape=4,
    interface="gymma",
    install="pip install matrix-games",
    tier="core",
    description="Matrix Games: 1-step coordination games (Prisoner's Dilemma, etc). Simplest testbed.",
    min_agents_for_baht=2,
    recommended_byzantine_budget=1,
    default_t_max=50_000,
    default_batch_size=64,
    default_buffer_size=64,
    default_batch_size_run=8,
    default_hidden_dim=16,
    obs_noise_std=0.0,
    action_noise_eps=0.0,
    env_kwargs={
        "key": "matrixgames:pdilemma-nostate-v0",
        "time_limit": 1,
    },
)


# ─── Extended Environments ────────────────────────────────────────────

OVERCOOKED = EnvConfig(
    name="overcooked",
    env_key="gymma",
    n_agents=2,
    n_actions=6,
    episode_limit=400,
    obs_shape=32,
    interface="gymma",
    install="pip install overcooked-ai",
    tier="extended",
    description="Overcooked: cooperative cooking. Only 2 agents — 1 Byzantine = fully adversarial.",
    min_agents_for_baht=2,
    recommended_byzantine_budget=1,
    default_t_max=1_000_000,
    default_batch_size=16,
    default_buffer_size=16,
    default_batch_size_run=2,
    default_hidden_dim=64,
    obs_noise_std=0.02,
    action_noise_eps=0.05,
    env_kwargs={
        "key": "overcooked:overcooked-cramped_room-v0",
        "time_limit": 400,
    },
)

SMAC_3S5Z = EnvConfig(
    name="smac-3s5z",
    env_key="sc2",
    n_agents=8,
    n_actions=14,
    episode_limit=150,
    obs_shape=128,
    interface="native",
    install="pip install pysc2; bash install_sc2.sh",
    tier="extended",
    description="SMAC 3s5z: 3 stalkers + 5 zealots vs AI. Large team, many possible Byzantines.",
    min_agents_for_baht=3,
    recommended_byzantine_budget=2,
    default_t_max=2_000_000,
    default_batch_size=32,
    default_buffer_size=32,
    default_batch_size_run=1,
    default_hidden_dim=64,
    obs_noise_std=0.02,
    action_noise_eps=0.05,
    env_kwargs={
        "map_name": "3s5z",
    },
)

SMAC_5V6 = EnvConfig(
    name="smac-5v6",
    env_key="sc2",
    n_agents=5,
    n_actions=12,
    episode_limit=70,
    obs_shape=98,
    interface="native",
    install="pip install pysc2; bash install_sc2.sh",
    tier="extended",
    description="SMAC 5m_vs_6m: 5 marines vs 6. Asymmetric, harder — Byzantine can throw the game.",
    min_agents_for_baht=3,
    recommended_byzantine_budget=1,
    default_t_max=2_000_000,
    default_batch_size=32,
    default_buffer_size=32,
    default_batch_size_run=1,
    default_hidden_dim=64,
    obs_noise_std=0.02,
    action_noise_eps=0.05,
    env_kwargs={
        "map_name": "5m_vs_6m",
    },
)


# ─── Registry ────────────────────────────────────────────────────────

ENVIRONMENTS: Dict[str, EnvConfig] = {
    # Core
    "mpe-pp": MPE_PP,
    "dsse": DSSE,
    "lbf": LBF,
    "advisor-trust": ADVISOR_TRUST,
    "matrix-games": MATRIX_GAMES,
    # Extended
    "overcooked": OVERCOOKED,
    "smac-3s5z": SMAC_3S5Z,
    "smac-5v6": SMAC_5V6,
}


def get_env_config(name: str) -> EnvConfig:
    """Get environment config by name."""
    if name not in ENVIRONMENTS:
        available = ", ".join(ENVIRONMENTS.keys())
        raise ValueError(f"Unknown environment '{name}'. Available: {available}")
    return ENVIRONMENTS[name]


def list_environments(tier: Optional[str] = None) -> List[EnvConfig]:
    """List all environments, optionally filtered by tier."""
    envs = list(ENVIRONMENTS.values())
    if tier:
        envs = [e for e in envs if e.tier == tier]
    return envs


def get_suite(suite_name: str) -> List[EnvConfig]:
    """Get a predefined suite of environments."""
    if suite_name == "core":
        return list_environments(tier="core")
    elif suite_name == "extended":
        return list_environments(tier="extended")
    elif suite_name == "all":
        return list_environments()
    elif suite_name == "quick":
        # Fast environments for smoke testing
        return [MATRIX_GAMES, ADVISOR_TRUST]
    elif suite_name == "standard":
        # The 3 main environments from prior work
        return [MPE_PP, DSSE, LBF]
    else:
        raise ValueError(f"Unknown suite '{suite_name}'. Options: core, extended, all, quick, standard")
