"""Matrix Games wrapper.

1-step coordination games (Prisoner's Dilemma, etc).
Simplest possible BAHT testbed — useful for debugging.

Install: pip install matrix-games
Uses GymmaWrapper for integration.
"""

# Matrix games are used via gymma.py with key="matrixgames:pdilemma-nostate-v0"
# No custom wrapper needed — the GymmaWrapper handles everything.
# This module exists for documentation and to register gym environments.

try:
    import matrixgames  # noqa: F401
    MATRIX_GAMES_AVAILABLE = True
except ImportError:
    MATRIX_GAMES_AVAILABLE = False
