# -*- coding: utf-8 -*-
"""
20 景区共享配置（跨场景论文 · 全量重跑）。

- name:      目录名 / cleaned CSV 前缀（data-project/cleaned_labeled_data/{name}_2bulu_cleaned.csv）
- en:        论文图/表英文名（恒山/衡山拼音相同，用 N/S 后缀区分）
- province:  省份
- geology:   地质类型缩写 G=花岗岩 V=火山岩 M=变质岩 D=丹霞 S=砂岩（公开地质知识初稿，待校核）
- climate:   气候带 subtropical=亚热带 warmtemperate=暖温带 temperate=温带
- data_dir:  原始爬取数据目录（data_20/<name>/）

bbox 由 derive_bbox_20.py 从 gpx 抽样推导，存 data-project/scenes_20_bbox.json。
"""

SCENES_20 = [
    {"name": "长白山",   "en": "Changbaishan", "province": "吉林", "geology": "V", "climate": "temperate"},
    {"name": "丹霞山",   "en": "Danxiashan",   "province": "广东", "geology": "D", "climate": "subtropical"},
    {"name": "峨眉山",   "en": "Emeishan",     "province": "四川", "geology": "V", "climate": "subtropical"},
    {"name": "梵净山",   "en": "Fanjingshan",  "province": "贵州", "geology": "M", "climate": "subtropical"},
    {"name": "恒山",     "en": "Hengshan-N",   "province": "山西", "geology": "M", "climate": "warmtemperate"},
    {"name": "衡山",     "en": "Hengshan-S",   "province": "湖南", "geology": "G", "climate": "subtropical"},
    {"name": "华山",     "en": "Huashan",      "province": "陕西", "geology": "G", "climate": "warmtemperate"},
    {"name": "黄山",     "en": "Huangshan",    "province": "安徽", "geology": "G", "climate": "subtropical"},
    {"name": "九华山",   "en": "Jiuhuashan",   "province": "安徽", "geology": "G", "climate": "subtropical"},
    {"name": "庐山",     "en": "Lushan",       "province": "江西", "geology": "M", "climate": "subtropical"},
    {"name": "青城山",   "en": "Qingcheng",    "province": "四川", "geology": "S", "climate": "subtropical"},
    {"name": "三清山",   "en": "Sanqingshan",  "province": "江西", "geology": "G", "climate": "subtropical"},
    {"name": "嵩山",     "en": "Songshan",     "province": "河南", "geology": "M", "climate": "warmtemperate"},
    {"name": "太白山",   "en": "Taibaishan",   "province": "陕西", "geology": "G", "climate": "warmtemperate"},
    {"name": "泰山",     "en": "Taishan",      "province": "山东", "geology": "M", "climate": "warmtemperate"},
    {"name": "五台山",   "en": "Wutaishan",    "province": "山西", "geology": "M", "climate": "temperate"},
    {"name": "武当山",   "en": "Wudangshan",   "province": "湖北", "geology": "M", "climate": "subtropical"},
    {"name": "武夷山",   "en": "Wuyishan",     "province": "福建", "geology": "D", "climate": "subtropical"},
    {"name": "雁荡山",   "en": "Yandangshan",  "province": "浙江", "geology": "V", "climate": "subtropical"},
    {"name": "张家界",   "en": "Zhangjiajie",  "province": "湖南", "geology": "S", "climate": "subtropical"},
]

NAMES_20 = [s["name"] for s in SCENES_20]
EN_20 = {s["name"]: s["en"] for s in SCENES_20}

# 大→小部署协议（阈值 1000 条清洗后轨迹）：
# 源池 = 12 大景区（>1000），目标池 = 8 小景区（<1000）
TARGET_SMALL = ["太白山", "衡山", "九华山", "丹霞山", "雁荡山", "武当山", "五台山", "恒山"]
TRAIN_BIG = [n for n in NAMES_20 if n not in TARGET_SMALL]
DATA_ROOT = "data_20"
CLEANED_DIR = "data-project/cleaned_labeled_data"
SEG_DIR = "data-project/segment_features"
BBOX_FILE = "data-project/scenes_20_bbox.json"


def scenes():
    """[{**属性, 'data_dir', 'gpx_dir', 'meta_csv'}]"""
    import os
    out = []
    for s in SCENES_20:
        d = dict(s)
        d["data_dir"] = os.path.join(DATA_ROOT, s["name"])
        d["gpx_dir"] = os.path.join(d["data_dir"], "gpx")
        d["meta_csv"] = os.path.join(d["data_dir"], "tracks_metadata.csv")
        out.append(d)
    return out
