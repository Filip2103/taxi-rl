"""
Sluzi da standardizuje rad sa okruzenjem Taxi-v3

Njegove uloge su:
 - kreiranje okruzenja
 - podesavanje seed-a
 - resetovanje epizode
 - izvrsavanje 1 akcije (step)
 - sakrivanje razlika izmedju gym i gymnasium

 Ovo je vazno jer ne vracaju isti format iz reset i step

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np


# Klasa koja cuva identifikator okruzenja. Frozen znaci da je objekat nepromenljiv nakon kreiranja
@dataclass(frozen=True)
class EnvSpec:
    env_id: str = "Taxi-v3"

# Funkcija za kreiranje okruzenja
def make_env(env_id: str = "Taxi-v3", seed: int = 0, render_mode: str | None = None):
    """
    Kreira Taxi okruženje (gym ili gymnasium) i postavlja početni seed.
    Ovaj sloj kompatibilnosti postoji kako ostatak koda ne bi morao da zna da li se koristi biblioteka gym ili gymnasium.
    """

    """
    gymnasium je novija verzija dok je gym stariji API
    """
    try:
        import gymnasium as gym  # type: ignore
    except Exception:
        import gym  # type: ignore

    # Kreiramo Taxi env. Ako je render mode zadat -> prosledi ga. Ukoliko nije onda napravi bez render mode
    # Ovo je vazno jer za trening i evaluaciju render obicno ne treba dk je za demo potreban render_mode="human"
    env = gym.make(env_id, render_mode=render_mode) if render_mode else gym.make(env_id)

    # Ovim se omogucuje kontrola ssed-a odnosno kontrolise pocetna stanja epizode i internu random logiku okruzenja.
    # Ako se koristi isti seed pocetne konfiguracije ce biti iste.
    try:
        env.reset(seed=seed)
    except TypeError:
        try:
            env.seed(seed)
        except Exception:
            pass
    
    # Ako bi negde korstio random akcije iz action space-a one bi bile deterministicke.
    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

    return env


"""
Funkcija sluzi za reset okruzenja na pocetno stanje i da zapocne novu epizodu.
Vraca pocetno stanje iz kog agent pocinje
Prima env i seed (opciono), a vraca stanje kao int
Svaka epizoda izgleda ovako:

reset -> stanje s0

agent bira akciju
env.step()

agent bira akciju
env.step()

...

dok epizoda ne zavrsi
"""
def reset_env(env, seed: int | None = None) -> int:
    # Ovo omogucava 2 rezima: obican reset ili deterministicki reset za specificnu epizodu
    if seed is not None:
        out = env.reset(seed=seed)
    else:
        out = env.reset()

    # Razlika izmedju gymnasium i gym-a
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
    else:
        obs = out
    # Posto je Taxi diskretno okruzenje stanje je 1 ceo broj
    return int(obs)

"""
Ova funkcija izvrsava jednu akciju agenta u okruzenju. To je 1 korak interakcije agent - okruzenje.
U svakom koraku agent uradi:

s → agent bira akciju a

env.step(a)

env vraća:
s'
r
done

To znaci: novo stanje, nagrada i da li je epizoda zavrsena

Na ovaj nacin razdvajamo pravi zavrsetak epizode i prekid zbog limita koraka. To je kljucno za ispravno racunanje sucess-rate-a.
Terminated - epizoda je zavrsena logikom zadatka
Truncated - epizoda je prekinuta zbog spoljnog ogranicenja (max_steps)
"""
    
def step_env(env, action: int) -> Tuple[int, float, bool, bool, dict]:
    out = env.step(int(action))
    # gymnasium: obs, reward, terminated, truncated, info
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return int(obs), float(reward), bool(terminated), bool(truncated), dict(info)

    # gym: obs, reward, done, info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        # gym nema posebno truncated i zbog toga ide false, a terminated done
        return int(obs), float(reward), bool(done), False, dict(info)

    raise RuntimeError(f"Unexpected env.step() output: {out}")

"""
Svaka RL epizoda izgleda ovako:

s = reset_env(env)

while not done:

    a = agent.act(s)

    s', r, terminated, truncated = step_env(env, a)

    done = terminated or truncated

    s = s'

"""