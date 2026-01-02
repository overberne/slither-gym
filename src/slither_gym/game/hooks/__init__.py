from importlib.resources import files

from slither_gym.game.enums import GameState


class GameHooks:
    ACT = (files(__package__) / 'act.js').read_text()
    GAME_STATE = (files(__package__) / 'game-state.js').read_text()
    GET_PLAYER_COUNT = (files(__package__) / 'get-player-count.js').read_text()
    GET_SERVER_IDS = (files(__package__) / 'get-server-ids.js').read_text()
    POLL_OBSERVATION = (files(__package__) / 'poll-observation.js').read_text()
    SET_SKIN = (files(__package__) / 'set-skin.js').read_text()
    SET_BOOST = (files(__package__) / 'set-boost.js').read_text()
    SET_HEADING = (files(__package__) / 'set-heading.js').read_text()
    SET_QUALITY = (files(__package__) / 'set-quality.js').read_text()
    ALL = '\n'.join(
        [
            (files(__package__) / 'act.js').read_text(),
            (files(__package__) / 'game-state.js').read_text(),
            (files(__package__) / 'get-player-count.js').read_text(),
            (files(__package__) / 'get-server-ids.js').read_text(),
            (files(__package__) / 'poll-observation.js').read_text(),
            (files(__package__) / 'set-skin.js').read_text(),
            (files(__package__) / 'set-boost.js').read_text(),
            (files(__package__) / 'set-heading.js').read_text(),
            (files(__package__) / 'set-quality.js').read_text(),
        ]
    )


class GameFunctions:
    GET_GAME_STATE = 'window.__getGameState()'
    GET_PLAYER_COUNT = 'window.__getPlayerCount()'
    GET_SERVER_IDS = 'window.__getServerIds()'
    POLL_OBSERVATION = 'window.__pollObservation()'
    SET_SKIN = 'window.__setSkin()'
    STOP_PLAYING = 'window.__resetGame()'

    @staticmethod
    def ACT(angle_rad: float, enable_boost: bool) -> str:
        return f'window.__act({angle_rad}, {'true' if enable_boost else 'false'})'

    @staticmethod
    def IS_GAME_IN_STATE(state: GameState) -> str:
        return f'window.__isGameInState("{state}")'

    @staticmethod
    def PLAY(nickname: str = 'beep boop') -> str:
        return f'window.__play("{nickname}")'

    @staticmethod
    def SET_HEADING(angle_rad: float) -> str:
        return f'window.__setHeading({angle_rad})'

    @staticmethod
    def SET_BOOST(enabled: bool) -> str:
        return f'window.__setBoost({'true' if enabled else 'false'})'

    @staticmethod
    def SET_QUALITY(high: bool) -> str:
        return f'window.__setQuality({'true' if high else 'false'})'

    @staticmethod
    def SET_SERVER(sid: int) -> str:
        """Only works if the game is at the login screen
        and a server with the specified id exists."""
        return f'window.__setServer("{sid}")'
