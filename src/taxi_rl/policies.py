from __future__ import annotations


from typing import Callable

import numpy as np

"""
Ovaj fajl sluzi da definise kako agent bira akcije. Ovaj fajl je most izmedju Q tabele i procesa donosenja odluke
Ovaj fajl ne uci nista, ne trenira nista, ne evaluira nista - samo definise pravila ponasanja agenta
Sadrzi funkcije koje vracaju polise odnosno funkcije za izbor akcije 

"""
"""
Ovo je funkcija koja pravi nasumičnu polisu.

Prima dva argumenta:

n_actions — broj mogućih akcija

rng — NumPy generator slučajnih brojeva

Vraća funkciju act(state) koja bira akciju. policy -> pokazuje na funkciju act

Ovo je dobro resenje jer bi u suprotnom morali da pitamo da li je random ili epsilon, a ovako imamo action = policy(state)


"""
def random_policy(n_actions: int, rng: np.random.Generator) -> Callable[[int], int]: # pi(s) = a (ulaz - stanje, izlaz - akcija)
    # Bira nasumican ceo broj 0 - n_actions-1. U Taxi-u n_actions=6
    # Ovde je _state samo formalno ulaz ali se ne koristi jer random polisa ne koristi stanje
    def act(_state: int) -> int:
        return int(rng.integers(0, n_actions))
    return act

"""
Epsilon greedy policy znaci:

sa verovatnocom e biras nasumicnu akciju (random action)
sa verovatnocom e-1 biras najbolju trenutno poznatu akciju (argmax Q(s,a))

Ovo resava problem exploration(istrazivanje novih akcija)/exploitation (koriscenje onoga sto agent zna)

E-greedy je vazan zato sto ako agent uvek bira samo najbolju poznatu akciju moze prerano da "zapne" u losoj strategiji
Ako agent uvek bira random onda nikad nece stabilno nauciti dobru startegiju

E-greedy daje balans izmedju ta 2

"""
def epsilon_greedy_policy(
    q_table: np.ndarray,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Callable[[int], int]: 
    # Ovde se koristi stanje za razliku od random polise
    def act(state: int) -> int:
        # Exploration deo gde ako je eps=0.1 onda u oko 10% slucajeva agent radi random potez.
        if rng.random() < float(epsilon):
            return int(rng.integers(0, n_actions))
        # Ako se ne desi exploration onda agent bira red iz Q tabele koji odgovara trenutnom stanju i akciju sa najvecom Q vrednoscu
        # Ako je q_table[state]= [1.2,0.5,2.7,...] onda np.argmax(q_table[state]) = 2 je je akcija 2 najbolja. To je iskoriscavanje naucenog znanja
        return int(np.argmax(q_table[state]))
    return act 