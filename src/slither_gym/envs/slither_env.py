import time
from typing import Any, Literal, SupportsFloat, TypedDict, cast

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray

from slither_gym.game import (
    DEFAULT_BASE_URL,
    Food,
    GameSession,
    GameState,
    MinimapSector,
    Snake,
    SnakeSegment,
)


class Observation(TypedDict):
    player_snake: Snake
    enemy_snakes: list[Snake]
    food: list[Food]
    minimap: list[MinimapSector]
    world_center: float
    world_radius: float
    time_since_last_decision: float


class Action(TypedDict):
    heading: NDArray[np.float32]
    boost: np.int32


class SlitherEnv(gym.Env[Observation, Action]):
    """
    Gymnasium environment wrapper for the Slither.io game.

    This environment runs a Slither.io game in a browser instance (via Playwright)
    and exposes a decision-driven interface where observations and actions are
    sampled at a configurable frequency. Decisions (observe -> act) are
    decoupled from rendering frames and are scheduled using a rolling
    decision boundary to keep timing consistent across steps.

    Observation structure (dict-like):
    - ``player_snake``: information about the controlled snake (`Snake` TypedDict).
    - ``enemy_snakes``: sequence of other snakes in the world (list of `Snake`).
    - ``food``: list of food items (list of `Food`).
    - ``minimap``: list of minimap sectors (list of `MinimapSector`), values are
        normalised to [-1, 1].
    - ``world_center``: single-value float for the world centre coordinate.
    - ``world_radius``: single-value float for the world radius.
    - ``time_since_last_decision``: float indicating elapsed time since the
        previous decision (useful because the observations are polled from a
        live game and may vary in latency).

    Action structure (dict-like):
    - ``heading``: one-dimensional numpy array containing the heading in radians
        (float32). The array shape is (1,).
    - ``boost``: discrete flag (0 or 1) indicating whether boost is enabled.

    The environment terminates when the controlled snake collides with the
    world boundary or another snake. Rewards are computed from the change
    in the in-game score, scaled by the ratio between the observed elapsed time
    and the configured decision interval.
    """

    metadata = {'render_modes': ['human']}

    _game_session: GameSession
    _nickname: str
    _current_score: int

    _decision_boundary: float
    _prev_decision_boundary: float
    _decision_interval: float
    _observation_runtime: float = 0

    def __init__(
        self,
        decision_frequency: float = 10.0,  # Hz
        nickname: str = 'beep boop',
        slither_url: str = DEFAULT_BASE_URL,
        render_mode: Literal['human'] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SlitherEnv.

        Parameters
        ----------
        decision_frequency : float, optional
            Decisions (observe -> act) per second. If ``0`` or ``np.inf`` is
            provided the decision interval is treated as zero (no delay), by
            default 10.
        nickname : str, optional
            Nickname to use when joining the game, by default 'beep boop'.
        slither_url : str, optional
            Base URL for the Slither.io game server. This is forwarded to the
            underlying ``GameSession`` constructor, by default
            ``DEFAULT_BASE_URL``.
        render_mode : Literal['human'] | None, optional
            When set to 'human' the browser runs with a visible window.
            Otherwise the browser is headless. By default this is ``None``.
        **kwargs : Any
            Additional keyword arguments are passed directly to
            ``slither_gym.game.GameSession``.
        """
        super().__init__()
        self._nickname = nickname
        # For keeping consistent decision timing.
        if decision_frequency == 0 or decision_frequency == np.inf:
            self._decision_interval = 0.0
        else:
            self._decision_interval = 1.0 / decision_frequency

        self._current_score = 0
        self._game_session = GameSession(
            headless=render_mode != 'human',
            slither_base_url=slither_url,
            **kwargs,
        )

        self.action_space = cast(
            spaces.Space[Action],
            spaces.Dict(
                {
                    'heading': spaces.Box(0.0, 2 * np.pi, (1,), dtype=np.float32),
                    'boost': spaces.Discrete(2, dtype=np.int32),
                }
            ),
        )
        self.observation_space = cast(
            spaces.Space[Observation],
            spaces.Dict(
                {
                    'player_snake': _new_snake_space(),
                    'enemy_snakes': spaces.Sequence(_new_snake_space()),
                    'food': _new_food_space(),
                    'minimap': _new_minimap_space(),
                    'world_center': spaces.Box(0, 2**31 - 1, (1,), dtype=np.int32),
                    'world_radius': spaces.Box(0, 2**31 - 1, (1,), dtype=np.int32),
                    'time_since_last_decision': spaces.Box(0, 10, (1,), dtype=np.int32),
                }
            ),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment and start (or restart) a game session.

        This call ensures the browser is on the login screen, starts playing
        with the provided nickname (from ``options['nickname']`` or the
        environment default), waits until the game enters the ``PLAYING``
        state and returns the initial observation. The environment sets an
        internal decision boundary which is used by :meth:`step` to schedule
        future observations at the configured decision frequency.

        Parameters
        ----------
        seed : int | None, optional
            Ignored. Present for Gym API compatibility.
        options : dict[str, Any] | None, optional
            Optional runtime options. Supported keys:
            - ``nickname``: str, nickname to use when joining the game.

        Returns
        -------
        tuple[Observation, dict[str, Any]]
            A tuple of the initial observation and an info dict. The info
            dict contains the following keys:
            - ``player_count``: int, number of players currently on the server.
        """
        nickname = options['nickname'] if options and 'nickname' in options else self._nickname

        if self._game_session.get_game_state() == GameState.PLAYING:
            self._game_session.stop_playing()

        self._game_session.wait_for_game_state(GameState.LOGIN_SCREEN)
        self._game_session.play(nickname=nickname)
        self._game_session.wait_for_game_state(GameState.PLAYING)
        player_count = self._game_session.get_player_count()
        game_data = self._game_session.poll_observation()
        obs = Observation(
            player_snake=game_data['player_snake'],
            enemy_snakes=game_data['enemy_snakes'],
            food=game_data['food'],
            minimap=game_data['minimap'],
            world_center=float(game_data['world_center']),
            world_radius=float(game_data['world_radius']),
            # Choose the theoretical interval as fill value
            time_since_last_decision=self._decision_interval,
        )

        self._prev_decision_boundary = time.time()
        self._decision_boundary = time.time() + self._decision_interval
        self._current_score = game_data['score']
        return obs, {'player_count': player_count}

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Apply an action and return the resulting transition.

        The provided action is sent to the game immediately. The environment
        then waits until the next decision boundary to poll a fresh
        observation; a rolling decision boundary is used to compensate for
        observation runtime so the effective decision frequency matches the
        configured ``decision_frequency`` as closely as possible.

        The returned observation includes ``time_since_last_decision`` which
        describes the elapsed time since the previous decision. Rewards are
        computed from the change in the in-game score, scaled by the ratio
        between the observed elapsed time and the configured decision
        interval.

        Parameters
        ----------
        action : Action
            Action to apply. Expected keys:
            - ``heading``: numpy array of shape (1,) with heading in radians.
            - ``boost``: discrete flag (0 or 1) to enable boosting.

        Returns
        -------
        tuple[Observation, float, bool, bool, dict[str, Any]]
            - ``observation``: Observation, fresh observation after applying the action.
            - ``reward``: float, reward computed from score delta normalized by time.
            - ``terminated``: bool, True if the episode ended (snake died or left world).
            - ``truncated``: bool, True if the episode was truncated (unused in this env).
            - ``info``: dict, additional runtime information (currently empty).
        """
        # Act
        self._game_session.act(action['heading'][0], action['boost'] == 1)

        # Delay observation to set consistent decision timing.
        time_until_observation = self._decision_boundary - time.time()
        if time_until_observation > 0:
            time.sleep(time_until_observation)

        # Observe
        decision_time = time.time()  # Start of (observe + act)
        game_data = self._game_session.poll_observation()
        time_since_last_decision = decision_time - self._prev_decision_boundary
        self._prev_decision_boundary = self._decision_boundary
        # Rolling decision boundary
        # - Use current time instead of boundary to determine next boundary.
        # - Subtract runtime to get actual frequency closer to frequency paraneter.
        self._decision_boundary = decision_time + self._decision_interval
        obs = Observation(
            player_snake=game_data['player_snake'],
            enemy_snakes=game_data['enemy_snakes'],
            food=game_data['food'],
            minimap=game_data['minimap'],
            world_center=float(game_data['world_center']),
            world_radius=float(game_data['world_radius']),
            time_since_last_decision=time_since_last_decision,
        )

        # Reward based on score increase, normalised by time delta.
        reward = (
            (game_data['score'] - self._current_score)
            * time_since_last_decision
            / self._decision_interval
        )
        # reward = transition['score'] - self._current_score
        self._current_score = game_data['score']
        return obs, reward, game_data['terminated'], False, {}

    def render(self) -> NDArray[np.int8] | list[NDArray[np.int8]] | None:
        # TODO: Render mode RGB via screenshots?
        return None

    def close(self) -> None:
        self._game_session.close()


def _new_segment_space(seed: int | None = None) -> spaces.Space[SnakeSegment]:
    return cast(
        spaces.Space[SnakeSegment],
        spaces.Dict(
            {
                'dx': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'dy': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            },
            seed=seed,
        ),
    )


def _new_snake_space(seed: int | None = None) -> spaces.Space[Snake]:
    return cast(
        spaces.Space[Snake],
        spaces.Dict(
            {
                'x': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'y': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'speed': spaces.Box(0.0, 14.0, (1,), dtype=np.float32),
                'heading': spaces.Box(0.0, 2 * np.pi, (1,), dtype=np.float32),
                'intended_heading': spaces.Box(0.0, 2 * np.pi, (1,), dtype=np.float32),
                'length': spaces.Box(0, 2**31 - 1, (1,), dtype=np.int32),
                'boosting': spaces.Discrete(2, dtype=np.int32),
                'segments': spaces.Sequence(_new_segment_space()),
            },
            seed=seed,
        ),
    )


def _new_food_space(seed: int | None = None) -> spaces.Space[Food]:
    return cast(
        spaces.Space[Food],
        spaces.Dict(
            {
                'x': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'y': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'size': spaces.Box(0, 2**31 - 1, (1,), dtype=np.int32),
            },
            seed=seed,
        ),
    )


def _new_minimap_space(seed: int | None = None) -> spaces.Space[MinimapSector]:
    return cast(
        spaces.Space[MinimapSector],
        spaces.Dict(
            {
                'x': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'y': spaces.Box(-1, 1, (1,), dtype=np.float32),
            },
            seed=seed,
        ),
    )


if __name__ == '__main__':
    env = SlitherEnv(render_mode='human', nickname='foo')
    env.reset()

    for i in range(360):
        obs, reward, terminated, truncated, info = env.step(
            Action(
                heading=np.array([np.deg2rad(np.float32(i * 10 % 360))]),
                boost=np.int32(0 if i > 180 else 1),
            )
        )

        if terminated:
            print(f'Terminated after {i} steps')
            break

        print(f'Reward: {reward}, delta time: {obs['time_since_last_decision']}')
