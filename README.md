# Slither Gym

Slither Gym is a small Gymnasium-compatible reinforcement learning environment
wrapping the Slither.io browser game. It exposes a decision-driven interface
that polls observations from a running game (via Playwright) and sends
discrete/continuous actions (heading + boost) back into the page.

The environment is intended for research and prototyping: it provides
high-level control over a single player snake and returns structured
observations as plain Python dictionaries (TypedDicts).

**NOTE:** This project drives a real browser instance using Playwright. You
must install the Playwright browsers after installing the package
(see Installation below).

## Quick Links

- **Code:** [src/slither_gym/slither_env.py](src/slither_gym/slither_env.py#L1)
- **Session controller:** [src/slither_gym/game/game_session.py](src/slither_gym/game/game_session.py#L1)
- **Hooks (JS):** [src/slither_gym/game/hooks](src/slither_gym/game/hooks)

## Installation

```bash
pip install -e .
# or install directly from the repository:
pip install git+https://github.com/overberne/slither-gym
```

- Install Playwright browsers (required):

```bash
python -m playwright install chromium
```

If you encounter Playwright issues on your platform, refer to the
Playwright documentation for platform-specific instructions.

## Features

- Gymnasium-compatible environment `SlitherEnv` with `reset()` / `step()`.
- Structured observations using TypedDicts (`player_snake`, `enemy_snakes`,
  `food`, `minimap`, `world_center`, `world_radius`, `time_since_last_decision`).
- Action space: continuous heading (radians) + discrete boost flag.
- Playwright-based `GameSession` exposing helpers to drive the in-page API.
- Small wrappers included for ergonomic action spaces (trigonometric action).

## Quickstart

Simple agent loop using the environment directly:

```python
import numpy as np
from slither_gym import SlitherEnv, Action

env = SlitherEnv(decision_frequency=10.0, nickname='agent', render_mode=None)
obs, info = env.reset()

for step in range(1000):
    # Keep heading straight (example) and no boost
    action = Action(heading=np.array([0.0], dtype=np.float32), boost=np.int32(0))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.close()
```

Using the trigonometric action wrapper, heading can be expressed as the sine and cosine of the angle:

```python
import numpy as np
from slither_gym import SlitherEnv
from slither_gym.wrappers.use_trigonometric_action import UseTrigonometricAction

env = UseTrigonometricAction(SlitherEnv(render_mode=None))
obs, info = env.reset()

angle = np.deg2rad(45.0)
sin, cos = np.sin(angle), np.cos(angle)
action = {'heading': np.array([sin, cos], dtype=np.float32), 'boost': np.int32(0)}
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## API Reference (Rudimentary)

- **Class:** `SlitherEnv` (see [src/slither_gym/slither_env.py](src/slither_gym/slither_env.py#L1))
  - Constructor parameters:
    - `decision_frequency: float = 10.0` — decisions per second. Use `0` or
      `np.inf` for zero-interval (no delay).
    - `nickname: str = 'beep boop'` — nickname used when joining the game.
    - `slither_url: str` — base URL for the Slither.io page (defaults to
      internal constant `DEFAULT_BASE_URL`).
    - `render_mode: 'human' | None` — when `'human'` the browser window is
      visible; otherwise browser is headless.
    - `**kwargs` forwarded to `GameSession` (e.g. `viewport`).

  - `reset(*, seed=None, options=None) -> tuple[Observation, dict]`
    - `options` supports `nickname` override.
    - Returns the initial `Observation` and an `info` dict containing
      `player_count`.

  - `step(action) -> tuple[Observation, float, bool, bool, dict]`
    - `action` expects a dict-like with:
      - `heading`: numpy array shape `(1,)` with heading in radians (float32).
      - `boost`: discrete flag `0` or `1` (int32).
    - Returns `(observation, reward, terminated, truncated, info)`.
    - Reward is computed from the change in in-game score and normalised
      by the configured decision interval.

  - `render()` — currently a stub (returns `None`).
  - `close()` — closes the underlying `GameSession`.

- **Observation structure** (see [src/slither_gym/game/types.py](src/slither_gym/game/types.py#L1))
  - `player_snake` (TypedDict `Snake`)
  - `enemy_snakes` (list of `Snake`)
  - `food` (list of `Food`)
  - `minimap` (list of `MinimapSector`) — normalised to [-1, 1]
  - `world_center`, `world_radius` (ints)
  - `time_since_last_decision` (float)

- **Action structure**
  - `heading`: numpy array with a single scalar in radians (shape `(1,)`)
  - `boost`: `0` or `1` (int)

- **GameSession** (Playwright controller — see [src/slither_gym/game/game_session.py](src/slither_gym/game/game_session.py#L1))
  - Constructor options: `headless`, `slither_base_url`, `viewport`.
  - Methods: `act(angle_rad, enable_boost)`, `set_heading(angle_rad)`,
    `set_boost(enabled)`, `poll_observation()`, `get_player_count()`,
    `get_game_state()`, `wait_for_game_state(state)`, `set_server(sid)`,
    `play(nickname)`, `stop_playing()`, `close()`.

## Package Hooks

The environment injects a set of small JavaScript helpers into the page to
expose a controlled in-page API. The JS files live under
[src/slither_gym/game/hooks](src/slither_gym/game/hooks) and are included as
package data. These hooks provide the bridge for `poll_observation()`,
`act()`, quality/skin settings and simple state queries.

## Development

- Run the example in `src/slither_gym/slither_env.py` by executing the file
  directly (it contains a minimal main-loop example). See the `if __name__ == '__main__'`
  blocks in `slither_env.py` and `game_session.py`.

- Run tests (if any are added in the future) with `pytest` from the project root.

## Known Limitations and Safety

- The environment drives a live browser—unreliable network or changes to the
  Slither.io front-end can break observation polling or hooks.
- The `render()` method is currently unimplemented; rendering is achieved
  by running with `render_mode='human'` which shows the real browser.

## Contributing

- Contributions are welcome. Please open issues or PRs for bugs, features,
  or documentation improvements.

## License

MIT License — see the LICENSE file.
