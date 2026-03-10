from __future__ import annotations

import argparse

from src.taxi_rl.env import make_env
from src.taxi_rl.utils import load_yaml
from src.taxi_rl.agents.q_learning import QLearningAgent
from src.taxi_rl.agents.sarsa import SARSAgent
from src.taxi_rl.online.trainer import OnlineTrainConfig, train_online

"""
Ovaj fajl predstavlja ulaznu tacku za online trening iz komandne linije

čita argumente iz terminala
učitava konfiguraciju
pravi env
pravi agenta
poziva trening
snima model

ovo je skripta koju korisnim direktno pokrece
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/taxi.yaml")
    p.add_argument("--agent", choices=["q_learning", "sarsa"], required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", required=True)
    args = p.parse_args()

    # ucitavanje konfiguracije koja se korisni kao osnovni izvor parametara, a komandna linija sluzi za override
    cfg = load_yaml(args.config)

    # ako ima env_id koristi se on, ako je default Taxi-v3
    env_id = cfg.get("env_id", "Taxi-v3")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    env = make_env(env_id, seed=seed)

    # Direktno se ucitava iz env-a
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    alpha = float(cfg.get("alpha", 0.1))
    gamma = float(cfg.get("gamma", 0.99))

    episodes = int(args.episodes if args.episodes is not None else cfg.get("episodes", 20000))

    train_cfg = OnlineTrainConfig(
        episodes=episodes,
        max_steps_per_episode=int(cfg.get("max_steps_per_episode", 200)),
        alpha=alpha,
        gamma=gamma,
        eps_start=float(cfg.get("eps_start", 1.0)),
        eps_end=float(cfg.get("eps_end", 0.05)),
        eps_decay=float(cfg.get("eps_decay", 0.9995)),
    )

    if args.agent == "q_learning":
        agent = QLearningAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
        train_online(env, agent, "q_learning", train_cfg, seed=seed)
        agent.save(args.save)
    else:
        agent = SARSAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
        train_online(env, agent, "sarsa", train_cfg, seed=seed)
        agent.save(args.save)

    print(f"Saved model -> {args.save}")


if __name__ == "__main__":
    main()


"""
Kompletan tok treninga:

train_online.py
    ↓
load_yaml()
    ↓
make_env()
    ↓
create agent
    ↓
train_online()
    ↓
agent.save()
"""