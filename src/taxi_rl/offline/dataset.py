from __future__ import annotations

from dataclasses import dataclass
from typing import  Optional

import pandas as pd


"""
Ovaj fajl predstavlja most izmedju collect_offline_dataset i offline/trainer
U sustini ovaj fajl kaze da ukoliko zelimo da offline trainer radi onda dataset mora izgledati ovako

definiše strukturu jedne tranzicije
omogućava čuvanje dataset-a u CSV fajl
omogućava učitavanje i proveru dataset-a iz CSV fajla

Offline trening radi sa tabelom tranzicija koja se definise u ovom fajlu.

"""

# 1 tranzicija predstavlja 1 korak interakcije (s,a,r,s'). U ovom projektu je to prosireno na (s,a,r,s',done,a')
# frozen true ide jer jednom zabelezno iskustvo iz okruzenje ne treba da se menja
@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: int # zbog CSV formata jer se cuva kao 0/1
    next_action: Optional[int] = None  # Samo za SARSA dodatak

# Cuva dataset u CSV fajl
def save_transitions_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

# Ucitava dataset i proverava da li ima odgovarajucu strukturu
def load_transitions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"state", "action", "reward", "next_state", "done"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["state"] = df["state"].astype(int)
    df["action"] = df["action"].astype(int)
    df["next_state"] = df["next_state"].astype(int)
    df["done"] = df["done"].astype(int)
    df["reward"] = df["reward"].astype(float)

    # Ako postoji ide u int, ako ne onda funkcija jos uvek radi. Na ovaj nacin isti loader moze da ucita oba algoritma.
    if "next_action" in df.columns:
        df["next_action"] = df["next_action"].astype(int)

    return df