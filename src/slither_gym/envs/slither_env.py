import time
from typing import Any, Literal, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from slither_gym.game import DEFAULT_BASE_URL, GameSession, GameState
from slither_gym.spaces import (
    Action,
    Food,
    MinimapSector,
    Observation,
    Slither,
    SlitherSegment,
    new_action_space,
    new_observation_space,
)

# Normalising constants
DISTANCE_NORM = PERCEPTION_RADIUS = np.float32(10_000)  # Used for normalising all distances
FOOD_SIZE_NORM = np.float32(50)
SLITHER_LENGTH_NORM = np.float32(50_000)
SPEED_NORM = np.float32(14)


class SlitherEnv(gym.Env[Observation, Action]):
    """
    Gymnasium environment wrapper for the Slither.io game.

    This environment controls a live Slither.io game instance running inside a
    browser (via Playwright) and exposes a decision-driven reinforcement learning
    interface. Environment steps are scheduled at a configurable decision
    frequency and are decoupled from the browser render loop to ensure consistent
    control timing under variable frame rates and network latency.

    Observations are sampled asynchronously from the running game and mapped
    into a structured, normalized representation suitable for learning agents.

    Observation
    -----------
    The observation is a dictionary-like structure with the following entries:

    ``player`` : Slither
        Observation of the controlled slither. Positions are expressed relative
        to the map center and normalized by the world radius.

    ``enemies`` : list[Slither]
        Observations of visible enemy slithers. Positions and motion quantities
        are expressed in the local reference frame of the player slither and
        normalized by the perception radius.

    ``food`` : list[Food]
        Visible food and prey entities expressed in the player’s local reference
        frame and normalized by the perception radius.

    ``minimap`` : list[MinimapSector]
        Coarse global spatial context indicating distant slither presence.
        Coordinates are normalized to the range ``[-1, 1]`` relative to the map
        center.

    ``nearest_border_heading_sin`` : ndarray, shape (1,)
        Sine of the heading angle from the player toward the nearest world border.

    ``nearest_border_heading_cos`` : ndarray, shape (1,)
        Cosine of the heading angle from the player toward the nearest world
        border.

    ``distance_to_border`` : ndarray, shape (1,)
        Distance to the nearest world border, normalized by the perception
        radius and capped at 1.

    ``time_since_last_decision`` : ndarray, shape (1,)
        Elapsed real time (in seconds) since the previous decision step. This is
        included because observations are sampled from a live game and may vary
        in latency.

    Action
    ------
    The action is a dictionary-like structure with the following entries:

    ``heading_sin`` : ndarray, shape (1,)
        Sine of the desired heading angle. Together with ``heading_cos``, this
        encodes the movement direction in a continuous, angle-invariant form.

    ``heading_cos`` : ndarray, shape (1,)
        Cosine of the desired heading angle.

    ``boost`` : ndarray, shape (1,)
        Binary flag indicating whether boost should be enabled (0 or 1).

    Episode Termination
    -------------------
    An episode terminates when the controlled slither dies, either due to
    collision with another slither or contact with the world boundary.

    Reward
    ------
    Rewards are computed from the change in the in-game score between decision steps.
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
        self.action_space = new_action_space()
        self.observation_space = new_observation_space()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
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

        # Choose the theoretical interval as fill value for time since last decision.
        obs = _map_observation(game_data, self._decision_interval)
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
            - ``heading_cos``: numpy array of shape (1,) with the cosine of
              the heading in radians.
            - ``heading_sin``: numpy array of shape (1,) with the sine of
              the heading in radians.
            - ``boost``: numpy array of shape (1,) with discrete flag (0 or 1)
              to enable boosting.

        Returns
        -------
        tuple[Observation, float, bool, bool, dict[str, Any]]
            - ``observation``: Observation, fresh observation after applying the action.
            - ``reward``: float, reward computed from score delta.
            - ``terminated``: bool, True if the episode ended (snake died or left world).
            - ``truncated``: bool, True if the episode was truncated (unused in this env).
            - ``info``: dict, additional runtime information (currently empty).
        """
        # Act
        self._game_session.act(
            angle_rad=np.atan2(action['heading_sin'][0], action['heading_cos'][0]),
            enable_boost=action['boost'][0] == 1,
        )

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
        obs = _map_observation(game_data, time_since_last_decision)

        # Reward based on score increase.
        reward = game_data['score'] - self._current_score
        self._current_score = game_data['score']
        return obs, reward, game_data['terminated'], False, {}

    def render(self) -> NDArray[np.int8] | list[NDArray[np.int8]] | None:
        # TODO: Render mode RGB via screenshots?
        return None

    def close(self) -> None:
        self._game_session.close()


def _map_observation(game_data: dict[str, Any], time_since_last_decision: float) -> Observation:
    world_center = float(game_data['world_center'])
    world_radius = float(game_data['world_radius'])
    player = game_data['player']
    player_x = float(player['xx'])
    player_y = float(player['yy'])
    player_x_world_center_ref = player_x - world_center
    player_y_world_center_ref = player_y - world_center
    distance_to_center = np.sqrt(
        player_x_world_center_ref**2 + player_y_world_center_ref**2, dtype=np.float32
    )
    nearest_border_cos = player_x_world_center_ref / distance_to_center
    nearest_border_sin = player_y_world_center_ref / distance_to_center
    distance_to_border = np.sqrt(
        (player_x_world_center_ref - world_radius) ** 2
        + (player_y_world_center_ref - world_radius) ** 2
    )
    distance_to_border = min(1, distance_to_border / PERCEPTION_RADIUS)
    player_heading_cos = np.cos([player['heading']], dtype=np.float32)
    player_heading_sin = np.sin([player['heading']], dtype=np.float32)

    return Observation(
        player=Slither(
            x=np.array([player_x_world_center_ref / world_radius], dtype=np.float32),
            y=np.array([player_x_world_center_ref / world_radius], dtype=np.float32),
            heading_cos=player_heading_cos,
            heading_sin=player_heading_sin,
            speed=np.array([player['speed'] / SPEED_NORM], dtype=np.float32),
            relative_heading_cos=np.ones((1,), dtype=np.float32),
            relative_heading_sin=np.zeros((1,), dtype=np.float32),
            relative_speed=np.zeros((1,), dtype=np.float32),
            length=np.array([player['length'] / SLITHER_LENGTH_NORM], dtype=np.float32),
            boosting=np.array([player['boosting']], dtype=np.int8),
            segments=_map_segment_batched(
                player['segments_x'],
                player['segments_y'],
                player_x,
                player_y,
                player_heading_cos[0],
                player_heading_sin[0],
            ),
        ),
        enemies=_map_enemies_batched(
            game_data['enemies'], player_x, player_y, player['heading'], player['speed']
        ),
        food=_map_food_batched(
            game_data['food'], player_x, player_y, player_heading_cos[0], player_heading_sin[0]
        ),
        minimap=[_map_minimap_sector(sector) for sector in game_data['minimap']],
        nearest_border_sin=np.array([nearest_border_sin], dtype=np.float32),
        nearest_border_cos=np.array([nearest_border_cos], dtype=np.float32),
        distance_to_border=np.array([distance_to_border], dtype=np.float32),
        time_since_last_decision=np.array([time_since_last_decision], dtype=np.float32),
    )


def _rotate_into_player_frame(
    dx: NDArray[np.float32],
    dy: NDArray[np.float32],
    heading_cos: float,
    heading_sin: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return cast(
        tuple[NDArray[np.float32], NDArray[np.float32]],  # Numpy promotion rules...
        (
            heading_cos * dx + heading_sin * dy,
            -heading_sin * dx + heading_cos * dy,
        ),
    )


def _map_enemies_batched(
    enemies: dict[str, list[Any]],
    player_x: float,
    player_y: float,
    player_heading: float,
    player_speed: float,
) -> list[Slither]:
    n = len(enemies["xx"])
    if n == 0:
        return []

    enemy_x = np.asarray(enemies["xx"], dtype=np.float32)
    enemy_y = np.asarray(enemies["yy"], dtype=np.float32)
    speed = np.asarray(enemies["speed"], dtype=np.float32)
    enemy_heading = np.asarray(enemies["heading"], dtype=np.float32)
    length = np.asarray(enemies["length"], dtype=np.float32)
    boosting = np.asarray(enemies["boosting"], dtype=np.int8)

    dx = enemy_x - player_x
    dy = enemy_y - player_y
    player_heading_cos = np.cos(player_heading, dtype=np.float32)
    player_heading_sin = np.sin(player_heading, dtype=np.float32)

    enemy_x_player_frame, enemy_y_player_frame = _rotate_into_player_frame(
        dx, dy, player_heading_cos, player_heading_sin
    )
    enemy_x_player_frame = enemy_x_player_frame / PERCEPTION_RADIUS
    enemy_y_player_frame = enemy_y_player_frame / PERCEPTION_RADIUS
    enemy_heading_cos = np.cos(enemy_heading, dtype=np.float32)
    enemy_heading_sin = np.sin(enemy_heading, dtype=np.float32)

    relative_heading = enemy_heading - player_heading
    relative_heading_cos = np.cos(relative_heading, dtype=np.float32)
    relative_heading_sin = np.sin(relative_heading, dtype=np.float32)
    relative_speed = (speed - player_speed) / SPEED_NORM
    speed = speed / SPEED_NORM

    out: list[Slither] = []
    for i in range(n):
        out.append(
            Slither(
                x=enemy_x_player_frame[i : i + 1],
                y=enemy_y_player_frame[i : i + 1],
                heading_cos=enemy_heading_cos[i : i + 1],
                heading_sin=enemy_heading_sin[i : i + 1],
                speed=speed[i : i + 1],
                relative_heading_cos=relative_heading_cos[i : i + 1],
                relative_heading_sin=relative_heading_sin[i : i + 1],
                relative_speed=relative_speed[i : i + 1],
                length=length[i : i + 1] / SLITHER_LENGTH_NORM,
                boosting=boosting[i : i + 1],
                segments=_map_segment_batched(
                    enemies['segments_x'][i],
                    enemies['segments_y'][i],
                    enemy_x[i],
                    enemy_y[i],
                    enemy_heading_cos[i],
                    enemy_heading_sin[i],
                ),
            )
        )
    return out


def _map_segment_batched(
    segments_x: list[float],
    segments_y: list[float],
    x: float,
    y: float,
    heading_cos: float,
    heading_sin: float,
) -> list[SlitherSegment]:
    n = len(segments_x)
    if n == 0:
        return []

    segments_x_np, segments_y_np = _rotate_into_player_frame(
        dx=(np.asarray(segments_x, dtype=np.float32) - x) / PERCEPTION_RADIUS,
        dy=(np.asarray(segments_y, dtype=np.float32) - y) / PERCEPTION_RADIUS,
        heading_cos=heading_cos,
        heading_sin=heading_sin,
    )
    return [
        SlitherSegment(
            x=segments_x_np[i : i + 1],
            y=segments_y_np[i : i + 1],
        )
        for i in range(n)
    ]


def _map_food_batched(
    food: dict[str, list[float]],
    player_x: float,
    player_y: float,
    player_heading_cos: float,
    player_heading_sin: float,
) -> list[Food]:
    n = len(food["xx"])
    if n == 0:
        return []

    food_x = np.asarray(food["xx"], dtype=np.float32)
    food_y = np.asarray(food["yy"], dtype=np.float32)
    size = np.asarray(food["size"], dtype=np.float32)

    dx = food_x - player_x
    dy = food_y - player_y
    food_x_player_frame = (player_heading_cos * dx + player_heading_sin * dy) / PERCEPTION_RADIUS
    food_y_player_frame = (-player_heading_sin * dx + player_heading_cos * dy) / PERCEPTION_RADIUS
    size = size / FOOD_SIZE_NORM

    return [
        Food(
            x=cast(NDArray[np.float32], food_x_player_frame[i : i + 1]),
            y=cast(NDArray[np.float32], food_y_player_frame[i : i + 1]),
            size=size[i : i + 1],
        )
        for i in range(n)
    ]


def _map_minimap_sector(sector: dict[str, Any]) -> MinimapSector:
    return MinimapSector(
        x=np.array([sector['x']], dtype=np.float32),
        y=np.array([sector['y']], dtype=np.float32),
    )


__all__ = [
    'SlitherEnv',
    'SPEED_NORM',
    'SLITHER_LENGTH_NORM',
    'DISTANCE_NORM',
    'PERCEPTION_RADIUS',
]

if __name__ == '__main__':
    env = SlitherEnv(render_mode='human', nickname='foo')
    env.reset()

    for i in range(360):
        action = env.action_space.sample()
        action['boost'][0] = 1
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f'Terminated after {i} steps')
            break

        print(f'Reward: {reward}, delta time: {obs['time_since_last_decision']}')
