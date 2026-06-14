#-----------------------------------------------------
"""
Rock Game State Helper 




"""
#-----------------------------------------------------

from functools import wraps


def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result

    return wrapper


def requires_money(cost):
    def decorator(func):
        @wraps(func)
        def wrapper(game, *args, **kwargs):
            if game.money < cost:
                raise ValueError(f"Not enough money. Need ${cost}, have ${game.money}.")

            game.money -= cost
            return func(game, *args, **kwargs)

        return wrapper

    return decorator












