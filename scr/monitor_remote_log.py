"""将远端项目任务日志的新增进度追加到本地 terminal.log。"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from project_config import CONFIG, PROJECT_ROOT


def valid_log_name(value: str) -> str:
    """限制监控对象为远端 logs 目录内的单个文件。"""

    candidate = Path(value)
    if candidate.name != value or value in {"", ".", ".."}:
        raise argparse.ArgumentTypeError("job_log 必须是 logs 目录内的单个文件名。")
    return value


def read_remote_log(job_log: str) -> list[str]:
    """通过配置的 SSH 别名读取远端项目日志末尾。"""

    monitor_config = CONFIG["monitoring"]
    remote_command = (
        f"cd {monitor_config['remote_project_dir']} && "
        f"tail -n {monitor_config['tail_lines']} "
        f"{monitor_config['remote_log_dir']}/{job_log}"
    )
    completed = subprocess.run(
        ["ssh", monitor_config["remote_ssh_alias"], remote_command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return [f"remote log read failed (exit={completed.returncode})"]
    return [line for line in completed.stdout.splitlines() if line]


def append_terminal_log(job_log: str, lines: list[str]) -> None:
    """将新的远端日志行以 UTC 时间戳追加到本地审计日志。"""

    terminal_log = PROJECT_ROOT / CONFIG["monitoring"]["local_terminal_log"]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with terminal_log.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(
                f"[{timestamp}] TERMINAL | remote-monitor {job_log} | {line}\n"
            )


def monitor(job_log: str, once: bool) -> None:
    """持续轮询远端日志；仅把从未写入过的行追加到 terminal.log。"""

    seen_lines: set[str] = set()
    poll_seconds = CONFIG["monitoring"]["poll_seconds"]
    while True:
        new_lines = [line for line in read_remote_log(job_log) if line not in seen_lines]
        if new_lines:
            append_terminal_log(job_log, new_lines)
            seen_lines.update(new_lines)
            for line in new_lines:
                print(f"[{job_log}] {line}", flush=True)
        if once:
            return
        time.sleep(poll_seconds)


def main() -> None:
    """解析作业日志名并启动单次或持续监控。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("job_log", type=valid_log_name)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    monitor(arguments.job_log, arguments.once)


if __name__ == "__main__":
    main()
