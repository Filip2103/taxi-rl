from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from src.taxi_rl.env import reset_env, step_env
from src.taxi_rl.policies import epsilon_greedy_policy
from src.taxi_rl.utils import EpsilonSchedule

"""
Ovaj fajl implementira online trening agenata

agent direktno komunicira sa okruzenjem
bira akcije
dobija nagradu
prelazi u novo stanje 
azurira Q tabelu

Ovaj fajl radi za oba algoritma
"""

AgentType = Literal["q_learning", "sarsa"]


@dataclass
class OnlineTrainConfig:
    episodes: int
    max_steps_per_episode: int
    alpha: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float

# Glavna funkcija fajla koja vraca listu ukupnih nagrada po epizodi
def train_online(env, agent, agent_type: AgentType, cfg: OnlineTrainConfig, seed: int = 0) -> List[float]:
    # Lokalni generator slucajnih brojeva za trening (isti seed daje isti tok slucajnih odluka)
    rng = np.random.default_rng(seed)
    # Kreira se objekat za eps schedule
    eps_sched = EpsilonSchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

    # lista nagrada gde se cuva zbir nagrada za svaku epizodu
    rewards: List[float] = []

    # spoljasnja petlja treninga gde je sveka iteracija 1 epizoda
    # U Taxi 1 epizoda traje od pocetnog stanja do uspesnog dropoff-a/max_broj koraka
    for ep in range(cfg.episodes):
        # reset i novo stanje s
        s = reset_env(env)
        # ukupna nagrada u toj epizodi
        ep_return = 0.0

        # racuna se eps za epizodu ep
        epsilon = eps_sched.value(ep)

        # pocetak za SARSA je drugaciji jer mora da zan pocetnu akciju vec na pocetku epizode
        if agent_type == "sarsa":
            act = epsilon_greedy_policy(agent.q, agent.n_actions, epsilon, rng)
            a = act(s)
        
        # Unutrasnja petlja po koracima
        for _t in range(cfg.max_steps_per_episode):
            # bira se akcija za q-learning u trenutnom stanju. To znaci da se konstruise eps greedy polisa i iz nje se dobija akcija a
            if agent_type == "q_learning":
                act = epsilon_greedy_policy(agent.q, agent.n_actions, epsilon, rng)
                a = act(s)

            # Ovde se izvrsava akcija, dobija novo stanje, nagrada i prima info da je epizoda zavrsena
            s_next, r, terminated, truncated, _info = step_env(env, a)
            # Za trening petlju epizoda je gotova u oba slucaja
            done = terminated or truncated
            ep_return += r

            # Update Q-learning
            if agent_type == "q_learning":
                agent.update(s, a, r, s_next, done)
                s = s_next
            else:
                # Update SARSA
                act2 = epsilon_greedy_policy(agent.q, agent.n_actions, epsilon, rng)
                a_next = act2(s_next)
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next

            if done:
                break
        # Cuvanje nagrade epizode
        rewards.append(ep_return)

    return rewards