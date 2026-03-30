#!/usr/bin/env python3
"""Demo runner: ARC-AGI-3 multigrid agent on a mock game.

Usage:
    python run.py                    # mock environment
    python run.py --toolkit          # official ARC-AGI Toolkit (requires arcagi package)
    python run.py --steps 20         # custom step count
"""

import argparse
import logging
import json
import sys

from agent import ArcMultigridAgent
from env_adapter import MockArcEnvironment, ToolkitArcEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run")


def run_game(agent: ArcMultigridAgent, env, max_steps: int = 50):
    """Run one game: observe-act loop."""
    obs, info = env.reset()
    agent.reset()
    logger.info("Game started. Grid shape: %s", obs.shape)

    total_reward = 0.0
    for step in range(max_steps):
        # Observe
        diag = agent.observe(obs, reward=total_reward, done=False, info=info)

        # Act
        action = agent.act()

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        logger.info(
            "Step %d: reward=%.3f total=%.3f kappa=%.3f done=%s",
            step, reward, total_reward, diag["kappa"], done,
        )

        if done:
            # Final observation
            agent.observe(obs, reward=reward, done=True, info=info)
            break

    summary = agent.summary()
    summary["game_over"] = done if 'done' in dir() else True
    summary["win"] = total_reward > 0.9 * max_steps
    return summary


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Multigrid Agent")
    parser.add_argument("--toolkit", action="store_true", help="Use official ARC-AGI Toolkit")
    parser.add_argument("--steps", type=int, default=10, help="Max steps per game")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size for mock env")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-file", type=str, default=None, help="Save log to JSON file")
    args = parser.parse_args()

    if args.toolkit:
        env = ToolkitArcEnvironment()
    else:
        env = MockArcEnvironment(grid_size=args.grid_size, num_steps=args.steps, seed=args.seed)

    agent = ArcMultigridAgent(default_grid_shape=(args.grid_size, args.grid_size))

    logger.info("Running ARC-AGI-3 Multigrid Agent")
    logger.info("Environment: %s", "Toolkit" if args.toolkit else "Mock")

    summary = run_game(agent, env, max_steps=args.steps)

    logger.info("=== Game Summary ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump({
                "summary": summary,
                "history": agent.log_history,
            }, f, indent=2, default=str)
        logger.info("Log saved to %s", args.log_file)

    env.close()
    return 0 if summary.get("game_over") else 1


if __name__ == "__main__":
    sys.exit(main())
