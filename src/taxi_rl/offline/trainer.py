from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd

# Funkcija podrzava oba algoritma
AgentType = Literal["q_learning", "sarsa"]

"""
Ovaj fajl implementira offline trening agenta
To znaci da agent ne poziva env.step(), ne bira akcije sam tokokm treninga, agent samo prolazi kroz vec postojece tranzicije iz dataset-a.
"""

# Konfiguracija za offline trening
@dataclass
class OfflineTrainConfig:
    epochs: int
    alpha: float
    gamma: float

# Glvna funkcija ovog fajla. Kao izlaz vraca prosecnu TD gresku po epohi. To omogucava da pratimo kako se trening menja kroz epohe, da li konvergira, da li se stabilizuje
def train_offline(agent, agent_type: AgentType, df: pd.DataFrame, cfg: OfflineTrainConfig, seed: int = 0) -> List[float]:
    # Mesanje redosleda tranzicija u svakoj epohi kako agent ne bi ucio uvek istim redosledm redova
    rng = np.random.default_rng(seed)
    # Cuva prosecnu apsolutnu promenu Q vrednosti
    td_errors: List[float] = []

    # Konverzija u numPy zbog brzine, lakseg indeksiranja i velikog broja tranzicija
    states = df["state"].to_numpy(dtype=np.int32)
    actions = df["action"].to_numpy(dtype=np.int32)
    rewards = df["reward"].to_numpy(dtype=np.float32)
    next_states = df["next_state"].to_numpy(dtype=np.int32)
    dones = df["done"].to_numpy(dtype=np.int8)

    # Poseban slucaj za SARSA jer mora imati a' u datasetu
    if agent_type == "sarsa":
        if "next_action" not in df.columns:
            raise ValueError("Offline SARSA requires 'next_action' column in dataset.")
        next_actions = df["next_action"].to_numpy(dtype=np.int32)
    else:
        next_actions = None  # Za Q-learning nepotrebna kolona

    # Niz indeksa gde N predstavlja broj tranzicija. Taj niz ce se kasnije mesati
    idx = np.arange(states.shape[0], dtype=np.int32)

    # Petlja po epohama. 1 epoha znaci 1 prolaz kro ceo dataset. Ako imas 400k tranzicija i 10 epoha to znaci da agent 10 puta prodje kroz 400k tranzicija
    for _ep in range(cfg.epochs):
        # Mesanje redosleda tranzicija
        rng.shuffle(idx)
        err_sum = 0.0
        
        # Glavna petlja kroz tranzicije
        for i in idx:
            s = int(states[i])
            a = int(actions[i])
            r = float(rewards[i])
            s_next = int(next_states[i])
            done = bool(dones[i])

            # Pre update-a se pamti stara vrednost da bi moglo da se vidi kolika je promena posle update-a
            old = float(agent.q[s, a])

            # Q-learning update
            if agent_type == "q_learning":
                agent.update(s, a, r, s_next, done)
            else:
                # SARSA update, ovde se iz dataset-a uzima a' i prosledjuje agentu
                a_next = int(next_actions[i])  
                agent.update(s, a, r, s_next, a_next, done)

            # Merenje promene Q vrednosti. Apsolutna prona te celije Q tabele odnosno koliko se Q[s,a] promeniko posle update-a. Ako je velika promena agent dosta uci ako nije onda se agent stabilizuje
            err_sum += abs(float(agent.q[s, a]) - old)

        # Prosecna greska po epohi. Ako opada kroz epohe - trening se stabilizuje, ako ostaje visok - model jos mnogo menja procene
        td_errors.append(err_sum / max(1, states.shape[0]))

    # Lista prosecnih promena po epohi
    return td_errors