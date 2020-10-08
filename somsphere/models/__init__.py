from enum import Enum

DY = 0.8660254


class Topology(Enum):
    GRID = "grid"
    SPHERE = "sphere"
    HEX = "hex"


class SomType(Enum):
    ONLINE = "online"
    BATCH = "batch"
