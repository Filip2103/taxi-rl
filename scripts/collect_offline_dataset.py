from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from src.taxi_rl.env import make_env, reset_env, step_env
from src.taxi_rl.offline.dataset import save_transitions_csv
from src.taxi_rl.agents.q_learning import QLearningAgent
from src.taxi_rl.agents.sarsa import SARSAgent

"""
Ovo je skripta za generisanje podataka u offline delu projekta.
Njegova uloga je da omoguci kontrolisano prikupljanje tranzicija iz okruzenja koriscenjem random i behaviour polise

Ovra skripta:

- pokrene Taxi v3 okruzenje
- pusta neku polisu da igra epizode
- belezi svaki korak kao tranziciju
- cuva sve tranzicije u CSV
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Taxi-v3")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--policy", choices=["random", "behavior"], default="random")
    p.add_argument("--model", default=None, help="Path to model .pkl (required if --policy behavior)")
    p.add_argument("--agent-type", choices=["q_learning", "sarsa"], default="q_learning",
                   help="What type of model is provided with --model (only used for behavior policy).")
    p.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for behavior epsilon-greedy policy.")

    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = make_env(args.env_id, seed=args.seed)
    n_actions = env.action_space.n
    rng = np.random.default_rng(args.seed)

    #  Ucitavanje behaviour modela
    model_q = None
    if args.policy == "behavior":
        if not args.model:
            raise ValueError("--model is required when --policy behavior")

        if args.agent_type == "q_learning":
            agent = QLearningAgent.load(args.model)
        else:
            agent = SARSAgent.load(args.model)

        # Nije potreban ceo agent za ponasanje vec samo Q tabela
        model_q = agent.q  # numpy array

    # Ova funkcija definise kako se bira akcija pri generisanju dataseta
    def act(state: int) -> int:
        # Ako se koristi random policy akcija je potpuno nasumicna
        if args.policy == "random" or model_q is None:
            return int(rng.integers(0, n_actions))

        # Ovo je behaviour polisa sto zanci da sa verovatnocom eps bira random akciju, a inace bira najbolju akciju prema Q tabeli modela
        if rng.random() < float(args.epsilon):
            return int(rng.integers(0, n_actions))
        return int(np.argmax(model_q[state]))

    # Lista tranzicija koja kasnije postaje tabela
    # Svaki element liste bice 1 dict sa kolonama (state,action,reward,next_State,done,next_action)
    rows = []
    # Petlja po epizodama, broj epizoda direktno utice na velicinu dataset-a i kolicinu podataka za offline trening
    for _ in range(args.episodes):
        # Pocetak epizode
        s = reset_env(env)
        done = False

        a = act(s)
        # Unutrasnja petlja po koracima
        for _t in range(args.max_steps):
            # Izvrsavanje akcije
            s_next, r, terminated, truncated, _info = step_env(env, a)
            done = terminated or truncated

            # Ako je epizoda zavrsena onda je a_next=0, u suprotnom biramo sledecu akciju
            if done:
                a_next = 0
            else:
                a_next = act(s_next)

            # Dodavanje tranzicije
            rows.append(
                {
                    "state": int(s),
                    "action": int(a),
                    "reward": float(r),
                    "next_state": int(s_next),
                    "done": int(done),
                    "next_action": int(a_next),
                }
            )

            # Kraj epizode
            if done:
                break

            s = s_next
            a = a_next

    df = pd.DataFrame(rows, columns=["state", "action", "reward", "next_state", "done", "next_action"])
    save_transitions_csv(df, args.out)
    print(f"Saved dataset with {len(df)} transitions -> {args.out}")
    print(f"Policy: {args.policy}" + (f" (epsilon={args.epsilon}, model={args.model})" if args.policy == "behavior" else ""))


if __name__ == "__main__":
    main()