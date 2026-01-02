from typing import TypedDict


class SnakeSegment(TypedDict):
    dx: float
    dy: float


class Snake(TypedDict):
    x: float
    y: float
    speed: float
    heading: float
    intended_heading: float
    length: int
    boosting: bool
    segments: list[SnakeSegment]


class Food(TypedDict):
    x: float
    y: float
    size: float


# Prey: Possible, but seems like low priority.


class MinimapSector(TypedDict):
    x: float
    y: float


class GameObservation(TypedDict):
    player_snake: Snake
    enemy_snakes: list[Snake]
    food: list[Food]
    minimap: list[MinimapSector]
    world_center: int
    world_radius: int
    score: int
    terminated: bool
