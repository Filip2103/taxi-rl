from __future__ import annotations

import argparse
import numpy as np

from src.taxi_rl.env import make_env, step_env
from src.taxi_rl.eval.evaluator import evaluate, EvalResult
from src.taxi_rl.agents.q_learning import QLearningAgent
from src.taxi_rl.agents.sarsa import SARSAgent



"""
Ova skripta omogucava da se iz terminala pokrene evaluacija.
Podrzava 2 rezima:

- standardna evaluacija kroz vise epizoda
- sweep evaluacija pocetnih stanja
"""


"""
Ova funkcija omogucava sweep evaluacija

- prolazi kroz sva moguca pocetna stanja
- za svako stanje pravi novi env
- rucno postavlja pocetno stanje Taxi okruzenja
- meri performanse modela iz svakog stanja
"""
def evaluate_start_state_sweep(env_id: str, seed: int, act_fn, max_steps: int = 200)-> EvalResult:
   
    # Privremeni env koji se pravi samo da bi se procitao broj stanja
    env0 = make_env(env_id, seed=seed)
    n_states = env0.observation_space.n
    try:
        env0.close()
    except Exception:
        pass

    returns = []
    steps_list = []
    successes = 0

    # Petlja kroz sva stanja
    for start_state in range(n_states):
        # Pravi se novi env za svako stanje
        env = make_env(env_id, seed=seed + start_state)
        _ = env.reset()  # Pozivamo da bi se wrapperi ispravno inicijalizovali. Time se spreva da se brojaci koraka, prethodna epizoda prenose izmedju evaluacija

        # Rucno postavljanje Taxi stanja (kljucna ideja sweep evaluacije)
        env.unwrapped.s = int(start_state)

        # Nakon sto se rucno postavi stnaje env-a i promenljiva s dobija istu vrednost
        s = int(start_state)
        ep_return = 0.0
        steps = 0

        # Petlja kroz korake epizoda
        for _ in range(max_steps):
            # Izbor akcije
            a = int(act_fn(s))
            s, r, terminated, truncated, _info = step_env(env, a)
            done = terminated or truncated
            ep_return += float(r)
            steps += 1
            if done:
                break

        try:
            env.close()
        except Exception:
            pass
        
        # Cuvanje rezultata
        returns.append(ep_return)
        steps_list.append(steps)
        if terminated:
            successes += 1

    arr_r = np.array(returns, dtype=np.float32)
    arr_s = np.array(steps_list, dtype=np.float32)

    return EvalResult(
        avg_reward=float(arr_r.mean()),
        std_reward=float(arr_r.std()),
        avg_steps=float(arr_s.mean()),
        success_rate=float(successes / n_states),
    )



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Taxi-v3")
    p.add_argument("--agent", choices=["random", "q_learning", "sarsa"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep-start-states", action="store_true")
    args = p.parse_args()

    env = make_env(args.env_id, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # Kreiranje polise koja potpuno nasumicno bira akcije
    if args.agent == "random":
        n_actions = env.action_space.n

        def act_fn(_s: int) -> int:
            return int(rng.integers(0, n_actions))
    
    # Ucitavas model iz Q tabele pravis greedy polisu. Evaluacija ne korsiti eps greedy nego a=argmaxQ(s,a)
    # To je standardno za evaluaciju
    elif args.agent == "q_learning":
        if not args.model:
            raise ValueError("--model is required for q_learning")
        agent = QLearningAgent.load(args.model)

        def act_fn(s: int) -> int:
            return int(np.argmax(agent.q[s]))

    else:  # Isto kao i kod Q-learning
        if not args.model:
            raise ValueError("--model is required for sarsa")
        agent = SARSAgent.load(args.model)

        def act_fn(s: int) -> int:
            return int(np.argmax(agent.q[s]))

    # Izbor rezima evaluacije
    if args.sweep_start_states:
        res = evaluate_start_state_sweep(
            args.env_id, args.seed, act_fn, max_steps=args.max_steps
        )
        print("=== Evaluation (START-STATE SWEEP) ===")
        print(f"Agent        : {args.agent}")
        if args.model:
            print(f"Model        : {args.model}")
        print(f"States       : {env.observation_space.n}")
        print(f"Avg reward   : {res.avg_reward:.2f}")
        print(f"Std reward   : {res.std_reward:.2f}")
        print(f"Avg steps    : {res.avg_steps:.2f}")
        print(f"Success rate : {res.success_rate:.2%}")
        return

    # Standardna evaluacija
    res = evaluate(env, act_fn, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)

    print("=== Evaluation ===")
    print(f"Agent        : {args.agent}")
    if args.model:
        print(f"Model        : {args.model}")
    print(f"Episodes     : {args.episodes}")
    print(f"Avg reward   : {res.avg_reward:.2f}")
    print(f"Std reward   : {res.std_reward:.2f}")
    print(f"Avg steps    : {res.avg_steps:.2f}")
    print(f"Success rate : {res.success_rate:.2%}")

if __name__ == "__main__":
    main()