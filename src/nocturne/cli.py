"""Command line interface for Nocturne research laboratory."""

import argparse
import sys


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="nocturne",
        description="Nocturne: Probabilistic learning of neural population dynamics"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # doctor
    subparsers.add_parser("doctor", help="Probe system environment and hardware capabilities")

    # synthetic
    synth_parser = subparsers.add_parser("synthetic", help="Manage controlled synthetic circuits")
    synth_sub = synth_parser.add_subparsers(dest="subcommand")
    gen_p = synth_sub.add_parser("generate", help="Generate synthetic trials")
    gen_p.add_argument("--all", action="store_true", help="Generate all 4 regimes")
    gen_p.add_argument("--trials", type=int, default=5000, help="Number of trials per regime")

    # train
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config YAML")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--config", type=str, help="Path to config YAML")
    eval_parser.add_argument("--nlb", action="store_true", help="Evaluate on NLB test sets")
    eval_parser.add_argument("--datasets", type=str, default="mc_maze,mc_rtt,area2_bump")

    # stream
    stream_parser = subparsers.add_parser("stream", help="Run online causal streaming")
    stream_parser.add_argument("--dataset", type=str, default="mc_maze_synthetic")
    stream_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")

    # geometry
    geom_parser = subparsers.add_parser("geometry", help="Latent geometry and fixed-point analysis")
    geom_parser.add_argument("--all", action="store_true")

    # data
    data_parser = subparsers.add_parser("data", help="Prepare datasets")
    data_parser.add_argument("--all", action="store_true")

    # demo
    subparsers.add_parser("demo", help="Run interactive demonstration")

    parsed = parser.parse_args(args)
    if not parsed.command:
        parser.print_help()
        return 0

    print(f"Nocturne CLI invoked: command={parsed.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
