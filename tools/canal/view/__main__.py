"""CLI entry point: canal-view <output.json> ... [options]"""

import argparse
import json
import sys
from pathlib import Path

from tools.canal.render import render_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canal-view",
        description="Pretty-print canal experiment results",
    )
    parser.add_argument("files", nargs="+", help="Experiment JSON output file(s)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include full artifact dump after summary",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate charts (not yet implemented)",
    )
    args = parser.parse_args()

    first = True
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        data = json.loads(path.read_text())
        if "analysis" not in data or "result" not in data:
            print(f"Not a canal output file: {path}", file=sys.stderr)
            continue

        if not first:
            print()
        print(render_summary(data, verbose=args.verbose))
        first = False


if __name__ == "__main__":
    main()
