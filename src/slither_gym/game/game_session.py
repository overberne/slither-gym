import time
import warnings
from typing import Any, TypedDict

from playwright.sync_api import Browser, Page, sync_playwright

from slither_gym.game.constants import DEFAULT_BASE_URL
from slither_gym.game.enums import GameState
from slither_gym.game.hooks import GameFunctions, GameHooks


class ViewportSize(TypedDict):
    width: int
    height: int


class GameSession:
    """
    Controller for a Playwright browser session running the Slither.io game.

    This class manages a Playwright browser, injects game hooks and provides
    helper methods to interact with the in-page game API. It exposes actions
    (heading, boost), observation polling and utility operations such as
    starting/stopping play and setting the server.

    The public methods wrap small JavaScript helpers (from
    ``slither_gym.game.hooks.GameFunctions``) and translate their results to
    Python-native types.
    """

    _is_browser_owned: bool = False
    _browser: Browser
    _page: Page

    def __init__(
        self,
        browser: Browser | None = None,
        headless: bool = True,
        slither_base_url: str = DEFAULT_BASE_URL,
        viewport: ViewportSize = {'width': 800, 'height': 600},
    ) -> None:
        """
        Start a Playwright browser and navigate to the Slither.io URL.

        Parameters
        ----------
        browser : playwright.Browser, optional
            An existing browser instance to use.
        headless : bool, optional
            Whether to run the browser in headless mode, by default ``True``.
        slither_base_url : str, optional
            The URL to load for the Slither.io game. By default this uses the
            module-level ``DEFAULT_BASE_URL``.
        viewport : ViewportSize, optional
            Initial viewport size passed to Playwright when creating the new
            page. Defaults to ``{'width': 800, 'height': 600}``.

        Notes
        -----
        The constructor starts Playwright, launches a Chromium browser and
        injects the JavaScript hooks required by the environment. It also
        configures some default game settings (skin and quality).
        """
        if browser is None:
            _playwright = sync_playwright().start()
            browser = _playwright.chromium.launch(
                headless=headless, args=['--disable-web-security']
            )
            self._browser = browser
            self._is_browser_owned = True

        self._page = browser.new_page(viewport=viewport)
        self._page.goto(slither_base_url)
        self._inject_hooks()
        self._page.evaluate(GameFunctions.SET_SKIN)
        self._page.evaluate(GameFunctions.SET_QUALITY(False))

    def close(self) -> None:
        self._page.close()
        if self._is_browser_owned and self._browser:
            self._browser.close()

    def act(self, angle_rad: float, enable_boost: bool) -> None:
        self._page.evaluate(GameFunctions.ACT(angle_rad, enable_boost))

    def set_heading(self, angle_rad: float) -> None:
        self._page.evaluate(GameFunctions.SET_HEADING(angle_rad))

    def set_boost(self, enabled: bool) -> None:
        self._page.evaluate(GameFunctions.SET_BOOST(enabled))

    def poll_observation(self) -> dict[str, Any]:
        return self._page.evaluate(GameFunctions.POLL_OBSERVATION)

    def get_player_count(self) -> int:
        return self._page.evaluate(GameFunctions.GET_PLAYER_COUNT)

    def get_game_state(self) -> GameState:
        try:
            state = self._page.evaluate("window.__getGameState()")
            if state in GameState:
                return GameState(state)
        except:
            pass
        return GameState.UNKNOWN

    def wait_for_game_state(self, state: GameState, timeout: float = 10.0) -> None:
        """Wait until the game's login UI is available or raise TimeoutError."""
        deadline = time.time() + float(timeout)
        while True:
            try:
                if self._page.evaluate(GameFunctions.IS_GAME_IN_STATE(state)):
                    return
            except Exception as e:
                warning = RuntimeWarning(*e.args)
                warning.with_traceback(e.__traceback__)
                print(warning)

            if time.time() > deadline:
                raise TimeoutError(f'Game state {state} did not appear within {timeout} seconds')

            time.sleep(0.1)

    def set_server(self, sid: int) -> None:
        """Set the desired server to a specific server id."""
        match self.get_game_state():
            case GameState.LOGIN_SCREEN:
                self._page.evaluate(GameFunctions.SET_SERVER(sid))
            case GameState.PLAYING:
                self.stop_playing()
                self.wait_for_game_state(GameState.LOGIN_SCREEN)
                self._page.evaluate(GameFunctions.SET_SERVER(sid))
            case _:
                raise RuntimeError('Cannot set server when game state is unknown.')

    def play(self, nickname: str = 'beep boop') -> None:
        match self.get_game_state():
            case GameState.LOGIN_SCREEN:
                self._page.evaluate(GameFunctions.PLAY(nickname))
            case GameState.PLAYING:
                warnings.warn('Already playing, ignoring play(name) call', RuntimeWarning)
            case _:
                raise RuntimeError('Cannot start play when not on the login screen')

    def stop_playing(self) -> None:
        match self.get_game_state():
            case GameState.LOGIN_SCREEN:
                warnings.warn(
                    'Already stopped playing, ignoring stop_playing() call', RuntimeWarning
                )
            case GameState.PLAYING:
                self._page.evaluate(GameFunctions.STOP_PLAYING)
            case _:
                raise RuntimeError('Cannot stop play when not in game')

    def _inject_hooks(self) -> None:
        self._page.evaluate(GameHooks.ALL)


if __name__ == '__main__':
    print('creating session')
    session = GameSession(headless=False)
    print('created')
    session.play('fooooooo')
    print('logged in')
    session.wait_for_game_state(GameState.PLAYING)
    print('done waiting for gamestate')
    session._page.wait_for_timeout(5000)  # type: ignore
    print('timeout 5000')
    session._page.evaluate(GameFunctions.STOP_PLAYING)  # type: ignore
    session._page.wait_for_timeout(5000)  # type: ignore
    session.play('baaaaaar')
    session._page.wait_for_timeout(10000)  # type: ignore
