from gymnasium.envs.registration import register  # pyright: ignore[reportUnknownVariableType]

from slither_gym.envs.slither_env import Action, Observation, SlitherEnv
from slither_gym.game import DEFAULT_BASE_URL

register(
    id='slither_gym/Slither-v0',
    entry_point='slither_gym.envs:SlitherEnv',
    nondeterministic=True,
    order_enforce=True,
)

__all__ = [
    'DEFAULT_BASE_URL',
    'Action',
    'Observation',
    'SlitherEnv',
]
