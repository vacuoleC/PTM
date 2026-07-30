"""安全解压并审计用户授权下载的 qPTM 原始归档。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from project_config import CONFIG, configured_path


def report_progress(message: str) -> None:
    """输出带时间戳的进度行，供远端日志监控器监督长操作。"""

    timestamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    """以分块读取计算归档 SHA-256，不把完整原始文件载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_path(root: Path, member: zipfile.ZipInfo) -> Path:
    """拒绝路径穿越和符号链接成员，返回安全的目标路径。"""

    if member.is_dir():
        return root / member.filename
    mode = member.external_attr >> 16
    if mode and (mode & 0o170000) == 0o120000:
        raise ValueError(f"qPTM 归档包含不允许的符号链接：{member.filename}")
    target = (root / member.filename).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"qPTM 归档包含路径穿越成员：{member.filename}")
    return target


def audit_and_extract(extract: bool) -> pd.DataFrame:
    """校验 ZIP 元数据，并按配置可选地安全解压全部成员。"""

    qptm_config = CONFIG["qptm"]
    if qptm_config["archive_format"] != "zip":
        raise ValueError("当前仅支持 config.yml 声明的 zip qPTM 归档格式。")
    archive_path = configured_path("qptm_archive")
    extract_root = configured_path("qptm_extract_dir")
    if not archive_path.is_file():
        raise FileNotFoundError(f"未找到 qPTM 归档：{archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(f"qPTM 文件不是有效 ZIP：{archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        files = [member for member in members if not member.is_dir()]
        report_progress(
            f"qPTM archive inspected; entries={len(members)}, files={len(files)}, "
            f"compressed_bytes={sum(member.compress_size for member in files)}"
        )
        if extract:
            extract_root.mkdir(parents=True, exist_ok=True)
            if any(extract_root.iterdir()) and not qptm_config["extract_overwrite"]:
                raise FileExistsError(
                    f"qPTM 解压目录非空，且 extract_overwrite=false：{extract_root}"
                )
            for position, member in enumerate(members, start=1):
                target = safe_member_path(extract_root, member)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if position % qptm_config["progress_every_entries"] == 0 or position == len(members):
                    report_progress(f"qPTM extraction completed {position} / {len(members)} entries")

    return pd.DataFrame(
        [
            {
                "archive_path": str(archive_path.relative_to(archive_path.parents[2])),
                "archive_sha256": sha256_file(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_entries": len(members),
                "archive_files": len(files),
                "compressed_bytes": sum(member.compress_size for member in files),
                "uncompressed_bytes": sum(member.file_size for member in files),
                "extract_requested": extract,
                "extract_directory": str(extract_root.relative_to(extract_root.parents[2])),
            }
        ]
    )


def main() -> None:
    """解析阶段参数、执行归档审计，并保存可版本化的摘要。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inspect", "extract"], default="extract")
    arguments = parser.parse_args()
    summary = audit_and_extract(extract=arguments.stage == "extract")
    output_path = configured_path("qptm_archive_audit")
    summary.to_csv(output_path, index=False)
    report_progress(f"qPTM archive audit completed; summary={output_path}")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
