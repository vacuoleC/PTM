"""CPTAC 的项目级初始化。

所有读取 CPTAC 数据的脚本都应先调用 :func:`configure_cptac`，以确保：

* 数据缓存到项目的 ``data/``，而非 Python 安装目录；
* 已下载但 MD5 与旧索引不一致的文件不会被删除后重下；
* 真正缺失的数据下载遇到临时网络错误时会自动重试。
"""

from __future__ import annotations

import os
import shutil
import time
import gzip
import zlib
from pathlib import Path

import cptac
import cptac.cancers.source as _source_module
import cptac.tools.download_tools as _download_module
from cptac.exceptions import HttpResponseError

from project_config import CONFIG, PROJECT_ROOT


def configure_cptac(project_root: Path | None = None) -> None:
    """将 cptac 配置为使用本项目的数据目录。

    该函数可以被重复调用；它不下载任何队列，真正读取某癌种时才会下载缺失文件。
    """
    root = project_root or PROJECT_ROOT
    root = root.resolve()
    data_dir = root / CONFIG["paths"]["data_dir"]
    data_dir.mkdir(exist_ok=True)

    # 如果用户此前已在 cptac 默认目录下载过小型索引或数据，则复用它们。
    default_data_dir = Path(cptac.__file__).resolve().parent / "data"
    if default_data_dir.is_dir():
        for source in default_data_dir.iterdir():
            destination = data_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif not destination.exists():
                shutil.copy2(source, destination)

    # cptac 的这些变量在模块导入时绑定，需要同时修改。
    cptac.CPTAC_BASE_DIR = str(root)
    _source_module.CPTAC_BASE_DIR = str(root)
    _download_module.DATA_DIR = str(data_dir)

    # cptac 导入时已启动后台版本检查；其原始 version() 会读取
    # CPTAC_BASE_DIR/version.py。上面将该目录改到项目后，需要让它仍从
    # 已安装包读取版本文件，否则后台线程会报无关的 FileNotFoundError。
    package_root = Path(cptac.__file__).resolve().parent

    def installed_cptac_version() -> str:
        namespace: dict[str, object] = {}
        exec((package_root / "version.py").read_text(encoding="utf-8"), namespace)
        return str(namespace["__version__"])

    cptac.version = installed_cptac_version

    def valid_gzip_cache(path: Path) -> bool:
        """完整读取 gzip 尾部，识别中断下载留下的损坏缓存。"""
        if path.suffix != ".gz":
            return True
        try:
            with gzip.open(path, "rb") as handle:
                while handle.read(CONFIG["cptac"]["gzip_validation_chunk_bytes"]):
                    pass
        except (EOFError, OSError, gzip.BadGzipFile, zlib.error):
            return False
        return True

    def safe_locate_files(self, datatype):
        """信任已有缓存；仅当文件缺失时下载，并处理临时网络错误。"""
        data_files = self.data_files[datatype]
        if not isinstance(data_files, list):
            data_files = [data_files]

        paths = []
        for data_file in data_files:
            cancer_type = (
                "all_cancers"
                if self.source in ("mssm", "harmonized")
                or (self.source == "washu" and datatype in ("tumor_purity", "hla_typing"))
                else self.cancer_type
            )
            dataset = f"{self.source}-{cancer_type}"
            file_path = data_dir / dataset / data_file

            cached_file_is_valid = file_path.is_file() and valid_gzip_cache(file_path)
            if not cached_file_is_valid and not self.no_internet:
                if file_path.is_file():
                    # 已验证损坏的缓存不可能被解析；删除后重新下载是可恢复操作。
                    print(f"发现损坏缓存，重新下载：{file_path.name}", flush=True)
                    file_path.unlink()
                last_error = None
                retry_attempts = CONFIG["cptac"]["download_retry_attempts"]
                for attempt in range(retry_attempts):
                    try:
                        cptac.download(self.cancer_type, self.source, datatype, data_file)
                        break
                    except HttpResponseError as error:
                        last_error = error
                        print(
                            f"下载 {data_file} 第 {attempt + 1}/{retry_attempts} 次失败：{error}",
                            flush=True,
                        )
                        if file_path.is_file():
                            file_path.unlink()
                        if attempt < retry_attempts - 1:
                            backoff = CONFIG["cptac"]["download_retry_backoff_seconds"]
                            time.sleep(backoff * (attempt + 1))
                else:
                    raise last_error

            paths.append(str(file_path))

        return paths[0] if len(paths) == 1 else paths

    _source_module.Source.locate_files = safe_locate_files
