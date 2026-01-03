"""MAMN-MARL: Multi-Agent Mobile Network + MEC (RLlib-ready)."""
"""MAMN-MARL: Multi-agent mobile network + MEC environment."""

__all__ = ["__version__"]

__version__ = "0.1.0"

from .env import MultiAgentMobileNetwork
from .rllib_entry import ENV_ID, rllib_env_creator, register_rllib_env
