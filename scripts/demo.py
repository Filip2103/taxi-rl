from __future__ import annotations

import argparse
import time
import numpy as np

from src.taxi_rl.env import make_env, reset_env, step_env
from src.taxi_rl.agents.q_learning import QLearningAgent
from src.taxi_rl.agents.sarsa import SARSAgent

"""
Ovaj fajl nam omogucava da se vizuelno proveri ponasanje istreniranog modela
Ako evaluate.py odgovara na pitanje koliko je dobar model, ova skripta odgovara na pitanje kako se model zaista ponasa korak po korak
Nekada metrike izgledaju dobro, a ponasanje nije intuitivno
Ovde se desava sledece:

- ucitavanje istreniranog modela
- pokretanje nekoliko epizoda u GUI rezimu
- prikaz ponasanja agenta korak po korak
- ispis osnovnih statistika po epizodi na kraju

"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Taxi-v3")
    p.add_argument("--agent", choices=["q_learning", "sarsa"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # Usporava korake da mozes da vidis sta agent radi, bez ovoga bi GUI isao prebrzo
    p.add_argument("--delay", type=float, default=0.05, help="seconds between steps (GUI)")
    p.add_argument("--pause-between", type=float, default=0.5, help="seconds between episodes")
    args = p.parse_args()

    # Kreiranje env-a sa GUI prikazom
    try:
        env = make_env(args.env_id, seed=args.seed, render_mode="human") 
    except TypeError:
        env = make_env(args.env_id, seed=args.seed)

    if args.agent == "q_learning":
        agent = QLearningAgent.load(args.model)
    else:
        agent = SARSAgent.load(args.model)

    # Ovde se koristi greedy polisa (agent u demou ne istrazuje, uvek bira najbolju akciju prema Q tabeli)
    # Ovde zelimo da vidimo sta je model stvarno naucio
    def act_fn(s: int) -> int:
        return int(np.argmax(agent.q[s]))

    # Statistike na kraju demoa
    returns = []
    steps_list = []

    # Petlja kroz epizode
    for ep in range(args.episodes):
        # Reset po epizodi sa razlitim seedom    
        s = reset_env(env, seed=args.seed + ep)

        ep_return = 0.0
        steps = 0
        done = False

        # Petlja kroz korake epizode
        for t in range(args.max_steps):
            # Izbor akcije i step
            a = act_fn(s)
            s, r, terminated, truncated, _info = step_env(env, a)
            done = terminated or truncated
            ep_return += float(r)
            steps += 1

            # Render + GUI osvezavanje (kratka pauza da se moze vizuelno pratiti kretanje)
            try:
                env.render()
            except Exception:
                pass
            # Bez sleep agent mi mozda preleteo epizodu pre nego sto stignes da vidis
            time.sleep(args.delay)

            # Ako agent zapadne u petlju ili epizoda traje dugo korisnik vidi da program nije zaledjen i zna koliko je koraka proslo
            if (t + 1) % 25 == 0:
                print(f"  [ep {ep+1}] step {t+1}/{args.max_steps} ...")

            if done:
                break
        # Ako agent ne uspe da zavrsi epizodu do limita to se eksplicitno prijavljuje
        if not done:
            print(f"  [ep {ep+1}] reached max_steps={args.max_steps} (forced stop)")

        returns.append(ep_return)
        steps_list.append(steps)
        success = bool(terminated)
        print(f"Episode {ep+1}/{args.episodes}: steps={steps}, return={ep_return:.2f}, success={success}")
        time.sleep(args.pause_between)

    r = np.array(returns, dtype=np.float32)
    st = np.array(steps_list, dtype=np.float32)

    print("\n=== Demo summary ===")
    print(f"Agent   : {args.agent}")
    print(f"Model   : {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Return  : mean={r.mean():.2f}, std={r.std():.2f}")
    print(f"Steps   : mean={st.mean():.2f}, std={st.std():.2f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()