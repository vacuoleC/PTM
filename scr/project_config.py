"""读取并验证项目唯一配置文件 ``config.yml``。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """加载 YAML 配置，并在文件缺失或为空时给出明确错误。"""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"找不到项目配置文件：{CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是非空 YAML 映射：{CONFIG_PATH}")
    return config


CONFIG = load_config()


def configured_path(name: str) -> Path:
    """将 ``paths`` 中的相对路径解析为项目内的绝对路径。"""
    try:
        configured = CONFIG["paths"][name]
    except KeyError as error:
        raise KeyError(f"config.yml 中缺少 paths.{name}") from error

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_template_path(name: str, **values: str) -> Path:
    """解析 ``paths`` 中带占位符的路径模板。"""
    try:
        template = CONFIG["paths"][name]
        configured = template.format(**values)
    except KeyError as error:
        raise KeyError(f"config.yml 中缺少 paths.{name} 或模板参数 {error}") from error

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_cohort_class(cptac_module: Any, cohort_name: str):
    """按配置中的 CPTAC 类名取得癌种类。"""
    try:
        class_name = CONFIG["datasets"]["cptac_class_names"][cohort_name]
    except KeyError as error:
        raise KeyError(f"config.yml 中未配置队列：{cohort_name}") from error

    return getattr(cptac_module, class_name)
