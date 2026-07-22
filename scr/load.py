"""
Phase 0 入口:加载 CPTAC 数据。

两步:
  1. 看 cptac 当前有哪些癌种可选。
  2. 实例化 LSCC(第一次会触发真正的数据下载,比索引文件大得多,
     可能要等几分钟;中途失败就重跑,这是 Zenodo 的老毛病,不是你的错)。

依赖:pip install cptac
"""

import os
import shutil
import time
import cptac


# ---------------------------------------------------------------------------
# (A) 把 cptac 的数据根目录从 conda 站点包内,改到本项目的 ./data
# ---------------------------------------------------------------------------
# 原理:cptac 1.5.14 把 base dir 写死在
#   - cptac/CPTAC_BASE_DIR  (cptac/__init__.py:21)
#   - cptac.tools.download_tools.DATA_DIR  (download_tools.py:13)
#   - cptac.cancers.source.CPTAC_BASE_DIR  (source.py 顶部 from import)
# 三处都是模块级"from ... import"绑定的字符串,但都进了模块命名空间。
# 下面在 import 完成后统一覆盖这三处,后续 locate_files / download 都会
# 走到 <project>/data 下。INDEX(cptac/__init__.py:72)在 import 时已读成
# DataFrame 常驻内存,本路径无关;只有 index.tsv 这个文件还需要在新位置
# 存在,初次运行会从 Zenodo 拉,所以新位置若缺 index.tsv,我们从 cptac
# 默认 data 复制一份(几千字节),免得 patch 后第一次没网。
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ_DATA = os.path.join(PROJ_ROOT, "data")
os.makedirs(PROJ_DATA, exist_ok=True)

# 一次性把 cptac 默认 data 目录里已下载好的文件搬过来(不删原位置,
# 留着当备份 / 调试用;已存在的同名文件不覆盖)。
_cptac_default_data = os.path.join(os.path.dirname(cptac.__file__), "data")
if os.path.isdir(_cptac_default_data):
    for name in os.listdir(_cptac_default_data):
        src = os.path.join(_cptac_default_data, name)
        dst = os.path.join(PROJ_DATA, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

# 改三处模块全局:之后所有 cptac 内部路径都基于 <project>/data
# cptac 内部用 os.path.join(CPTAC_BASE_DIR, "data/{dataset}/{file}") 拼路径,
# 所以 CPTAC_BASE_DIR 必须是 <project> 根目录(不是 data 子目录),才能让
# 拼出来的实际路径落在 <project>/data/{dataset}/。
# download_tools.DATA_DIR 则是 download() 自己用的根,需要直接指到 data。
import cptac.cancers.source as _src_mod
import cptac.tools.download_tools as _dt_mod

cptac.CPTAC_BASE_DIR = PROJ_ROOT
_src_mod.CPTAC_BASE_DIR = PROJ_ROOT
_dt_mod.DATA_DIR = PROJ_DATA
os.makedirs(_dt_mod.DATA_DIR, exist_ok=True)
print(f"[cptac patch] CPTAC_BASE_DIR={PROJ_ROOT}; DATA_DIR={PROJ_DATA}", flush=True)


# ---------------------------------------------------------------------------
# (B) 跳过 cptac 自带的"md5 不一致就删掉 93MB 重下"逻辑 + 缺文件时重试
# ---------------------------------------------------------------------------
# 背景:cptac 1.5.14 INDEX 里硬编码的 checksum 是 Zenodo 的旧版本;Zenodo 上
# 同一份文件的新版本 md5 改了 → 每次调用 cptac.Lscc() 都会:
#   1) 算本地 md5;
#   2) 对不上 INDEX 里的旧 checksum;
#   3) warn + os.remove(93MB 文件);
#   4) 重新下 93MB → 还要撞 504。
# 这是在"自残"已知的好文件。下面这个 patch 让 locate_files 直接信任本地
# 缓存,只对真正缺失的文件去 Zenodo 拉,而且撞 504 / 超时时重试 3 次。

from cptac.exceptions import HttpResponseError


def _safe_locate(self, datatype):
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
        file_path = os.path.join(cptac.CPTAC_BASE_DIR, f"data/{dataset}/{data_file}")

        if not os.path.isfile(file_path) and not self.no_internet:
            # 缺文件:去 Zenodo 拉,撞 504 / 网抽风时重试 3 次
            ok = False
            last_err: Exception = HttpResponseError("placeholder")
            for attempt in range(3):
                try:
                    cptac.download(self.cancer_type, self.source, datatype, data_file)
                    ok = True
                    break
                except HttpResponseError as e:
                    last_err = e
                    print(f"[cptac patch] download {data_file} attempt {attempt + 1}/3 failed: {e}",
                          flush=True)
                    if os.path.isfile(file_path):  # 删不完整文件,下次重试从零开始
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
            if not ok:
                raise last_err  # 3 次都失败:抛干净错给上层

        paths.append(file_path)

    return paths[0] if len(paths) == 1 else paths


_src_mod.Source.locate_files = _safe_locate
print("[cptac patch] installed: skip md5-redownload + retry-3 on Zenodo 504", flush=True)


# 1. 看有哪些癌种可选。
#    cptac 提供的等价 API:
#      - cptac.list_datasets()                    -> 完整 (Cancer, Source, Datatype) 三列表
#      - cptac.get_cancer_options()               -> 按癌种聚合,返回 Cancer -> [Source/Datatype]
#    旧名字 list_cancer_options() 不存在(Pylance 报错正解)。
print(cptac.get_cancer_options())


# 2. 加载 LSCC。
#    首次实例化会去 Zenodo 拉数据,文件较大;失败就重跑。
lscc = cptac.Lscc()

ph = lscc.get_phosphoproteomics("umich")

 # 有行有列 = 数据真下来了
print(ph.shape)

# 打得出 .N 样本 = 正常样本约定成立
print([s for s in ph.index.astype(str) if s.endswith(".N")][:5])
