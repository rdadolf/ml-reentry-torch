"""CLI entry point: canal <config.py> [options]"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canal",
        description="torch.compile analysis tool",
    )
    parser.add_argument("config", help="Python config file defining EXPERIMENTS")
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory (default: canal_output/<timestamp>)",
    )
    parser.add_argument("--only", help="Run only the named experiment")
    parser.add_argument(
        "--list", action="store_true", help="List experiments without running"
    )
    args = parser.parse_args()

    from tools.canal.config import load_config

    experiments = load_config(args.config)

    if args.list:
        for exp in experiments:
            models = ", ".join(
                m if isinstance(m, str) else getattr(m, "__name__", "<fn>")
                for m in exp.models
            )
            print(f"  {exp.name:30s}  {exp.analysis:10s}  {models}")
        return

    if args.only:
        experiments = [e for e in experiments if e.name == args.only]
        if not experiments:
            print(f"No experiment named {args.only!r}", file=sys.stderr)
            sys.exit(1)

    from tools.canal.runner import run_all

    print(f"canal: running {len(experiments)} experiment(s)")
    output_dir = run_all(experiments, args.output_dir)
    print(f"canal: results in {output_dir}")


if __name__ == "__main__":
    main()
