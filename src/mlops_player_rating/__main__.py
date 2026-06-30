"""Entry point so the project can be executed with ``python -m mlops_player_rating``."""

import sys
from pathlib import Path
from typing import Any


def _find_run_params() -> dict[str, Any]:
    return {}


def main(*args, **kwargs) -> None:
    from kedro.framework.cli.utils import find_run_command

    package_name = Path(__file__).parent.name
    run = find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main(sys.argv[1:])
