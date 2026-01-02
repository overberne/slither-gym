from enum import StrEnum, auto


class GameState(StrEnum):
    LOGIN_SCREEN = auto()
    PLAYING = auto()
    UNKNOWN = auto()
