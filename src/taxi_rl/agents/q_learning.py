from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.taxi_rl.utils import load_pickle, save_pickle

"""
Fajl implementira Q-learning agenta koji:

    čuva Q tabelu
    ažurira vrednosti tokom treninga
    može da sačuva model
    može da učita model

Ovaj fajl implementira algoritam ucenja dok trainer.py upravlja epizodama, policies.py bira akcije, a env.py komunicira sa okruzenjem.
"""

# U Taxi-v3 Q tabela ima dimenzije 500x6 (500 stanja, 6 akcija)
@dataclass
class QLearningAgent:
    n_states: int
    n_actions: int
    alpha: float
    gamma: float

    # Ovde se kreira Q tabela. Q tabela je matrica Q(s,a)
    def __post_init__(self):
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    # Ovo je glavna funkcija algoritma. 
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        # Ovde se racuna max Q(s',a). To je najveca Q vrednost u sledecem stanju
        # Ovo predstavlja definiciju off-policy algoritma jer agent uzima maksimalnu vrednost, a ne vrednost akcije koju je stvarno izabrao.
        best_next = float(np.max(self.q[s_next]))
        """
        Ovde je implementirana Bellmanova jednacina. 
        target = r + gamma*maxQ(s',a)
        Ako je done=True nema buducih nagrade i onda je target = r potpuno ispravno.
        """
        target = r + (0.0 if done else self.gamma * best_next)
        """
        Q update: Q(s,a)<- Q(s,a) + alpha(target - Q(s,a))
        Primer: Q(s,a) = 2, target = 6,alpha = 0.1 -> Q(s,a) = 2.4 (agent polako uci)
        """
        self.q[s, a] = self.q[s, a] + self.alpha * (target - self.q[s, a])

    # Cuvanje modela (q tabela i meta informacije: n_statesn_actions,alpha,gamma)
    def save(self, path: str) -> None:
        save_pickle({"q": self.q, "meta": self.__dict__}, path)

    # Ucitava model. Umesto da se samo ucita Q tabela, pravi se kompletan agent sto je koristno za evaluaciju
    @staticmethod
    def load(path: str) -> "QLearningAgent":
        obj = load_pickle(path)
        meta = obj["meta"]
        agent = QLearningAgent(
            n_states=meta["n_states"],
            n_actions=meta["n_actions"],
            alpha=meta["alpha"],
            gamma=meta["gamma"],
        )
        agent.q = obj["q"]
        return agent
    

    """
    Tok treninga:

    agent = QLearningAgent(...)
    policy = epsilon_greedy_policy(...)

    s = reset_env()
    a = policy(s)
    s', r, done = step_env()
    agent.update(s, a, r, s')
    s = s
    """