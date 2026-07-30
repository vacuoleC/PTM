"""Read-only monitor for the known PTMv2 E2.2 remote OOF task."""
import argparse
import subprocess
from pathlib import Path

import yaml


def parse_status(text: str) -> dict[str, str]:
    """Parse tab-separated status fields emitted by the fixed remote command."""
    result = {}
    for line in text.splitlines():
        if "\t" in line:
            key, value = line.split("\t", maxsplit=1)
            result[key] = value
    return result


def remote_status(monitoring: dict) -> dict[str, str]:
    """Return process, progress, and output status without modifying the server."""
    pid_file = monitoring["e2_2_smoke_pid_file"]
    log_file = monitoring["e2_2_smoke_log_file"]
    output_file = monitoring["e2_2_smoke_output_file"]
    command = " ".join(
        [
            f"pid=$(tr -d '\\r\\n' < '{pid_file}'); printf 'pid\\t%s\\n' \"$pid\";",
            f"if ps -p \"$pid\" -o etime=,stat=,pcpu=,pmem=,cmd= > /tmp/e2_2_ps.$$; then printf 'process\\t'; tr '\\n' ' ' < /tmp/e2_2_ps.$$; printf '\\n'; rm -f /tmp/e2_2_ps.$$; else printf 'process\\tNOT_RUNNING\\n'; fi;",
            f"printf 'progress\\t'; grep -F '[E2.2]' '{log_file}' | tail -n 1 || true; printf '\\n';",
            f"if test -f '{output_file}'; then printf 'output_lines\\t'; wc -l < '{output_file}'; printf 'output_sha256\\t'; sha256sum '{output_file}' | awk '{{print $1}}'; else printf 'output_lines\\tABSENT\\n'; fi",
        ]
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", monitoring["remote_ssh_alias"], command],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_status(completed.stdout)


def main(config_path: Path) -> None:
    """Load monitor settings and print a stable local monitoring summary."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = remote_status(config["monitoring"])
    for key in ("pid", "process", "progress", "output_lines", "output_sha256"):
        if key in status:
            print(f"{key}={status[key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml"
    )
    main(parser.parse_args().config)
