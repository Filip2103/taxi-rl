from __future__ import annotations

import argparse

from src.taxi_rl.env import make_env
from src.taxi_rl.utils import load_yaml
from src.taxi_rl.offline.dataset import load_transitions_csv
from src.taxi_rl.offline.trainer import OfflineTrainConfig, train_offline
from src.taxi_rl.agents.q_learning import QLearningAgent
from src.taxi_rl.agents.sarsa import SARSAgent

"""
Ovaj fajl ucitava podatke, pravi odgovarajuceg agenta, pokrece offline trening i cuva istrenirani model
Ovaj fajl predstavlja glavnu ulaznu skriptu za offline ucenje
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/taxi.yaml")
    p.add_argument("--agent", choices=["q_learning", "sarsa"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)

    # Env ovde ne sluzi isto kao i kod online treninga. 
    # Ovde env sluzi da ocita broj stanja i broj akcija
    env_id = cfg.get("env_id", "Taxi-v3")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    env = make_env(env_id, seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    alpha = float(cfg.get("alpha", 0.1))
    gamma = float(cfg.get("gamma", 0.99))

    df = load_transitions_csv(args.dataset)

    off_cfg = OfflineTrainConfig(epochs=int(args.epochs), alpha=alpha, gamma=gamma)


    # Pravi se odgovarajuci agent, pokrece offline trening i cuva se model. Agent u startu ima praznu Q tabelu, trening je onda popunjava i azurira kroz dataset
    if args.agent == "q_learning":
        agent = QLearningAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
        train_offline(agent, "q_learning", df, off_cfg, seed=seed)
        agent.save(args.save)
    else:
        # Ako dataset nema next_action train_offline ce baciti gresku za SARSA
        agent = SARSAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
        train_offline(agent, "sarsa", df, off_cfg, seed=seed)
        agent.save(args.save)

    print(f"Saved model -> {args.save}")


if __name__ == "__main__":
    main()