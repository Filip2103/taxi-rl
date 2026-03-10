from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.taxi_rl.utils import load_pickle, save_pickle

"""
U ovom fajlu je implementiran SARSA agent cija je uloga da:

    čuva Q tabelu
    ažurira vrednosti tokom treninga
    može da sačuva model
    može da učita model

"""

@dataclass
class SARSAgent:
    n_states: int
    n_actions: int
    alpha: float
    gamma: float

    # Q-tabela
    def __post_init__(self):
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    # Kljucna razliku u odnosu na Q-learning. U ulazu imamo i a_next - sledecu akciju
    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        # Za razliku od Q-learning gde uzimamo najbolju mogucu akcju u sledecem stanju, ovde uzimamo akciju koju ce agent stvarno sledecu izabrati.
        # Umesto maxQ(s',a') koristi se Q(s',a'), a' - konkretna sledeca akcija. AKo je s_next=10, a_next=3 -> Q[10,3]
        next_q = float(self.q[s_next, a_next])
        # target = r + gamma*Q(s',a'). Ako je epizoda zavrsena onda nema buduce vrednosti i target samo r (kao kod Q-learning)
        target = r + (0.0 if done else self.gamma * next_q)
        # Standardni TD update: Q(s,a) <- Q(s,a) + alpha(target - Q(s,a)). Isto kao i Q-learning, razlika samo kako se target racuna
        self.q[s, a] = self.q[s, a] + self.alpha * (target - self.q[s, a])

    # Cuva model (Q tabela, meta informacije)
    def save(self, path: str) -> None:
        save_pickle({"q": self.q, "meta": self.__dict__}, path)

    # Ucitava model
    @staticmethod
    def load(path: str) -> "SARSAgent":
        obj = load_pickle(path)
        meta = obj["meta"]
        agent = SARSAgent(
            n_states=meta["n_states"],
            n_actions=meta["n_actions"],
            alpha=meta["alpha"],
            gamma=meta["gamma"],
        )
        agent.q = obj["q"]
        return agent