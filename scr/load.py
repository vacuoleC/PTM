"""
Phase 0 入口:加载 CPTAC 数据。

两步:
  1. 看 cptac 当前有哪些癌种可选。
  2. 实例化 LSCC(第一次会触发真正的数据下载,比索引文件大得多,
     可能要等几分钟;中途失败就重跑,这是 Zenodo 的老毛病,不是你的错)。

依赖:pip install cptac
"""

import cptac


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