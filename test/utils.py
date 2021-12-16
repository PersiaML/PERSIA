import random


def random_port(start: int = 10000, end: int = 65535) -> int:
    return random.randint(start, end)
