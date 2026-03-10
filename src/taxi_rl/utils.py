from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml

"""
Ovaj fajl ima 4 glavne funkcije:

- ucitavanje konfiguracije
- rad sa fajlovima 
- cuvanje i ucitavanje modela
- epsilon decay schedule
"""

#Ova funkcija otvara fajl, parsira YAML i vraca Python dictionary
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Ova funkcija osigurava da direktorijum postoji
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# Ova f-ja cuva objekat u fajl. Cuva q-tabelu i agent objekat
def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Ova f-ja ucitava model i q tabelu
def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# Ovaj objekat se koristi u online treningu. Izracuna se novi epsilon, nova e-greedy polisa se napravi i agent koristitu polisu tokom epizode.
# Na pocetku trening aagent skoro uvek bira random akcije (eps=1), dok kasnije skoro uvek bira najbolju akciju
@dataclass
class EpsilonSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.9995

    def value(self, episode_idx: int) -> float:
        # Eps decaj je pozeljan jer agent brzo prestaje da bude potpuno random ali zadrzava malo istrazivanja
        eps = self.eps_start * (self.eps_decay ** episode_idx)
        return float(max(self.eps_end, eps))


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)