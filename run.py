#!/usr/bin/env python3
"""ARC-AGI-3 multigrid agent runner.

Usage:
    python run.py                              # mock environment
    python run.py --game ls20                  # official ARC-AGI Toolkit
    python run.py --game ls20 --render terminal
    python run.py --mock --steps 20            # mock with custom steps
"""

import argparse
import logging
import json
import sys

from agent import ArcMultigridAgent
from env_adapter import (
    MockArcEnvironment, ToolkitArcEnvironment,
    GameStatus, Action,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run")


def run_game(agent: ArcMultigridAgent, env, max_steps: int = 200):
    """Run one game: observe-act loop."""
    obs = env.reset()
    agent.reset()
    logger.info("Game started. Grid shape: %s, win_levels: %d",
                obs.grid.shape, obs.win_levels)

    for step in range(max_steps):
        diag = agent.observe(obs)

        if obs.status != GameStatus.PLAYING:
            logger.info("Game ended: %s at step %d", obs.status.value, step)
            break

        action = agent.act()
        obs = env.step(action)

        logger.info(
            "Step %d: action=%d levels=%d/%d kappa=%.3f status=%s",
            step, action.action_id, obs.levels_completed,
            obs.win_levels, diag["kappa"], obs.status.value,
        )

    summary = agent.summary()
    summary["game_over"] = obs.status != GameStatus.PLAYING
    summary["win"] = obs.status == GameStatus.WIN
    return summary


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Multigrid Agent")
    parser.add_argument("--game", type=str, default=None,
                        help="Game ID for ARC-AGI Toolkit (e.g. ls20)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock environment (default if no --game)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max steps per game")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Grid size for mock env")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", type=str, default="terminal",
                        help="Render mode: terminal, terminal-fast, human, none")
    parser.add_argument("--api-key", type=str, default=None,
                        help="ARC API key (or set ARC_API_KEY env var)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save log to JSON file")
    args = parser.parse_args()

    use_toolkit = args.game is not None and not args.mock

    if use_toolkit:
        render = args.render if args.render != "none" else None
        env = ToolkitArcEnvironment(
            game_id=args.game,
            seed=args.seed,
            render_mode=render,
            api_key=args.api_key,
        )
        logger.info("Using ARC-AGI Toolkit: game=%s", args.game)
    else:
        env = MockArcEnvironment(
            grid_size=args.grid_size,
            num_steps=args.steps,
            seed=args.seed,
        )
        logger.info("Using mock environment: grid=%d, steps=%d", args.grid_size, args.steps)

    agent = ArcMultigridAgent()
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
    return 0 if summary.get("win") else 1


if __name__ == "__main__":
    sys.exit(main())
