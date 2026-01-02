from typing import Any, SupportsFloat, cast

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from slither_gym import Action, Observation, SlitherEnv


class UseTrigonometricAction(gym.ActionWrapper[Observation, Action, Action]):
    def __init__(self, env: SlitherEnv):
        super().__init__(env)
        self.action_space = cast(
            spaces.Space[Action],
            spaces.Dict(
                {
                    'heading': spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                    'boost': spaces.Discrete(2, dtype=np.int32),
                }
            ),
        )

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # action['heading'].shape == (2,)
        sin, cos = action['heading'][:, None]  # Keep array dims for scalar sin and cos

        action = action.copy()
        action['heading'] = np.atan2(sin, cos)
        return self.env.step(action)


if __name__ == '__main__':
    env = UseTrigonometricAction(SlitherEnv(render_mode='human', nickname='foo'))
    env.reset()

    for i in range(360):
        angle = np.deg2rad(np.float32(i * 10 % 360))
        sin, cos = np.sin(angle), np.cos(angle)
        obs, reward, terminated, truncated, info = env.step(
            Action(
                heading=np.array([sin, cos]),
                boost=np.int32(0 if i > 180 else 1),
            )
        )

        if terminated:
            print(f'Terminated after {i} steps')
            break

        print(f'Reward: {reward}, delta time: {obs['time_since_last_decision']}')
