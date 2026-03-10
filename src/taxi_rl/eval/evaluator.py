from __future__ import annotations

from dataclasses import dataclass
from typing import  List

import numpy as np

from src.taxi_rl.env import reset_env, step_env


"""
Ovaj fajl implementira funkciju koja:

- pokrece vise epizoda u okruzenju
- koristi zadatu polisu akcije
- belezi statistiku
- vraca metrike

Ova funkcija meri u sustini koliko je dobar vec istreniran model
Ovde se ne trenira model vec se samo mere performanse
"""

@dataclass
class EvalResult:
    avg_reward: float # prosecna nagrada po epizode  
    std_reward: float # standardna devijacija nagrade (mala devijacija - stabilna polisa, velika devijacija - nestabilno ponasanje)
    avg_steps: float # prosecan broj koraka do zavrsetka epizode (manji broj koraka znaci efikasnije resavanje zadatka)
    success_rate: float # procenat uspesnih epizoda


def evaluate(env, act_fn, episodes: int, max_steps: int = 200, seed: int = 0) -> EvalResult:
    returns: List[float] = [] # ukupna nagrada svake epizode
    steps_list: List[int] = [] # broj koraka svake epizode
    successes: int = 0 # broj epizoda koje su zavrsene uspesnim dropoff-om

    # petlja kroz epizode (agent samo izvrsava akcije)
    for _ in range(episodes):
        # ovim se radi reeseed env u svakoj epizodi kako bi se dobila raznolikost pocetnih stanja
        try:
            _ = env.reset(seed=seed + _) # ep1 -> seed 0, ep2 -> seed 1...
            s = reset_env(env)
        except TypeError:
            s = reset_env(env)
        ep_return = 0.0
        steps = 0
        done = False  

        # Petlja kroz korake epizode
        for _t in range(max_steps):
            # Izbor akcije. Evaluator ne zna nista o agentu on dobije funkciju act_fn(state) -> action
            # Na ovaj nacin je obezbedjeno da evaluator moze raditi sa Q-leraning agentom, SARSA, random, i bilo kojom drugom polisom
            a = int(act_fn(s))
            s, r, terminated, truncated, _info = step_env(env, a)
            # terminated - uspesan dropoff
            # truncated - epizoda prekinuta zbog limita koraka
            done = terminated or truncated
            ep_return += r
            steps += 1
            if done:
                break
        # Cuvanje statistike epizode
        returns.append(ep_return)
        steps_list.append(steps)

        # Nacin na koji merimo success je preko terminated zato sto jedino sto se smatra uspesnim jeste kada imamo dropoff uspesan
        if terminated:
            successes += 1

    arr = np.array(returns, dtype=np.float32)
    st = np.array(steps_list, dtype=np.float32)
    
    # Ovde se racuna finalna statistika
    return EvalResult(
        avg_reward=float(arr.mean()),
        std_reward=float(arr.std()),
        avg_steps=float(st.mean()),
        success_rate=float(successes / episodes),
    )