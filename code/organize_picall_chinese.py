"""统一整理结果图件目录、中文文件名和 SVG 可编辑文字。

本脚本只处理结果图件，不处理原始影像、GeoTIFF 数据或表格。
对无法由词典稳定翻译的英文文件名，使用“图件+数字编号”，
同时在根目录保存映射表，避免丢失来源追溯。
"""

from __future__ import annotations

import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}


COLLECTIONS = [
    (WORKSPACE / "geocoding/results/picall/主流程", WORKSPACE / "geocoding/results/picall/主流程"),
    (WORKSPACE / "geocoding/results/picall/注册复现", WORKSPACE / "geocoding/results/picall/注册复现"),
    (WORKSPACE / "geocoding/results/paper_projection", WORKSPACE / "geocoding/results/picall/论文投影"),
    (WORKSPACE / "geocoding/results/thesis_fig_5_9_5_10", WORKSPACE / "geocoding/results/picall/论文图件"),
    (WORKSPACE / "geocoding/siwei_image1_volume_geocode/PICALL", WORKSPACE / "geocoding/siwei_image1_volume_geocode/results/picall"),
    (WORKSPACE / "projection_correction/touying/results", WORKSPACE / "projection_correction/touying/results/picall"),
    (
        WORKSPACE / "projection_correction/touying_roof_workflow/results/paper_projection",
        WORKSPACE / "projection_correction/touying_roof_workflow/results/picall/论文投影",
    ),
    (
        WORKSPACE / "projection_correction/touying_roof_workflow/results/blue_aligned",
        WORKSPACE / "projection_correction/touying_roof_workflow/results/picall/蓝色边界对齐",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/results/picall/正式图件",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/results/picall/正式图件",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/results/picall/过程图件",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/results/picall/过程图件",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall/正式图件",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall/正式图件",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall/过程图件",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall/过程图件",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_roof_only/results/picall/正式图件",
        WORKSPACE / "height_estimation/building_height_estimation_roof_only/results/picall/正式图件",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_roof_only/results/picall/过程图件",
        WORKSPACE / "height_estimation/building_height_estimation_roof_only/results/picall/历史过程",
    ),
    (
        WORKSPACE / "geocoding/results/outputs/pic_all2_reproduction/unused_ppt",
        WORKSPACE / "geocoding/results/picall/归档/未采用演示图",
    ),
    (
        WORKSPACE / "height_estimation/ps_triangle_height_estimation/results/height_estimation_analysis_ready_ps/figures",
        WORKSPACE / "height_estimation/ps_triangle_height_estimation/results/picall/分析就绪结果",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/clean_workflow/02_base_projection",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall/清晰流程/底面投影",
    ),
    (
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/output/figures/PICALL_individual_png",
        WORKSPACE / "output/像素偏移建筑高度估计PICALL结果/picall",
    ),
    (
        WORKSPACE / "output/像素偏移建筑高度估计PICALL结果/SVG图片",
        WORKSPACE / "output/像素偏移建筑高度估计PICALL结果/picall",
    ),
    (
        WORKSPACE / "output/同济校区PS三角面建筑估高方法报告/SVG图片",
        WORKSPACE / "output/同济校区PS三角面建筑估高方法报告/picall",
    ),
]


PHRASES = {
    "pixel_offset_height_map": "像素偏移建筑高度图",
    "prior_projection_sar_correction": "先验投影与合成孔径雷达校正",
    "pixel_offset_quality_audit": "像素偏移质量审计",
    "pixel_offset_single_building_diagnostic": "像素偏移单体建筑诊断",
    "all_buildings_elevation0m_projection": "全部建筑零米高程投影",
    "all_buildings_shp_height_projection": "全部建筑矢量高度投影",
    "shp_height_local_sar_correction": "矢量高度局部雷达校正",
    "sar_building_feature_enhancement": "合成孔径雷达建筑特征增强",
    "shape_adaptive_enhanced_sar_correction": "形态自适应增强雷达校正",
    "hybrid_shape_adaptive_local_correction": "混合形态自适应局部校正",
    "hybrid_pixel_offset_building_height_map": "混合像素偏移建筑高度图",
    "image_feature_only_local_registration": "纯影像特征局部配准",
    "image_feature_only_height_map": "纯影像特征建筑高度图",
    "image_feature_only_registration_audit": "纯影像特征配准审计",
    "joint_quantity_quality_registration": "数量质量联合配准",
    "joint_quantity_quality_height_map": "数量质量联合建筑高度图",
    "roof_only_full_area_height_map": "仅屋顶全区建筑高度图",
    "roof_only_height_summary": "仅屋顶建筑高度汇总",
    "roof_only_vs_strict_joint_comparison": "仅屋顶与严格联合方法对比",
    "roof_only_height_search_process": "仅屋顶高度搜索过程",
    "roof_only_diagnostic": "仅屋顶估高诊断",
    "full_area_projection_blue_aligned_overlay": "全区投影蓝色边界对齐叠加图",
    "full_area_projection_corrected_overlay": "全区校正投影叠加图",
    "full_area_projection_overlay": "全区投影叠加图",
    "dsm_simulated_sar_vs_real": "数字表面模型模拟雷达与实测对比",
    "registered_strict_triangle_projection": "配准后严格三角面投影",
    "registered_strict_triangle_refined_mask": "配准后严格三角面精化掩膜",
    "tongji_sar_intensity_with_buildings": "同济校区雷达强度与建筑轮廓",
    "sar_overview_selected_buildings_pixels": "雷达总览与选定建筑像素",
    "initial_projection_masks": "初始投影掩膜",
    "per_building_sar_crops": "逐建筑雷达裁剪图",
    "refined_masks": "精化掩膜",
    "geographic_same_pixel_comparison": "同像素地理定位对比",
    "initial_vs_refined_masks": "初始与精化掩膜对比",
    "method_geocoded_points": "建筑约束地理编码点",
    "gamma_vs_proposed_map": "伽马软件与建筑约束方法对比图",
    "3d_scatter_points": "三维散射点",
    "zoomed_geocoded_points": "地理编码点局部放大",
    "same_ps_pixel_3d_buildings": "同像素永久散射体三维建筑",
    "same_ps_pixel_radar_pixels": "同像素永久散射体雷达像素",
    "absolute_height_roof_projection": "绝对高程屋顶投影",
    "registered_roof_projection": "配准后屋顶投影",
    "roof_anchored_side_triangle_projection": "屋顶锚定侧墙三角面投影",
    "ps_roof_wall_assignment": "永久散射体屋顶墙面归属",
    "full_area_triangle_projection": "全区三角面投影",
    "full_area_ps_quality_assessment": "全区永久散射体质量评估",
    "ps_height_plan_overlay_buildings": "永久散射体高度与建筑平面叠加图",
    "highrise_optimized_building_height_estimation": "高层优化建筑高度估计",
    "local_amplitude_mask_refinement": "局部幅度掩膜精化",
    "full_area_mask_refinement_overview": "全区掩膜精化总览",
    "triangle_projection_ps_fusion": "三角面投影与永久散射体融合",
    "highrise_optimization_validation": "高层优化验证",
    "highrise_optimized_building_height_map": "高层优化建筑高度图",
}


TEXT_REPLACEMENTS = [
    ("Tongji SAR intensity with building footprints", "同济校区合成孔径雷达强度与建筑轮廓"),
    ("Building vector projection optimization in SAR coordinates", "合成孔径雷达坐标中的建筑矢量投影优化"),
    ("Range Pixel (1 m grid)", "距离向像素（1米网格）"),
    ("Azimuth Pixel (1 m grid)", "方位向像素（1米网格）"),
    ("Range distance / m", "距离向距离／米"),
    ("Azimuth distance / m", "方位向距离／米"),
    ("Range column", "距离向列号"),
    ("Azimuth row", "方位向行号"),
    ("Range pixel", "距离向像素"),
    ("Azimuth pixel", "方位向像素"),
    ("Longitude / deg", "经度／度"),
    ("Latitude / deg", "纬度／度"),
    ("Longitude", "经度"),
    ("Latitude", "纬度"),
    ("Easting / m (UTM 51N)", "东向坐标／米（通用横轴墨卡托51北带）"),
    ("Northing / m (UTM 51N)", "北向坐标／米（通用横轴墨卡托51北带）"),
    ("Easting (m, UTM zone 51N)", "东向坐标／米（通用横轴墨卡托51北带）"),
    ("Northing (m, UTM zone 51N)", "北向坐标／米（通用横轴墨卡托51北带）"),
    ("Height / m", "高度／米"),
    ("Height (m)", "高度（米）"),
    ("Building count", "建筑数量"),
    ("PS count", "永久散射体数量"),
    ("Building constrained", "建筑约束方法"),
    ("GAMMA/DEM", "伽马软件／数字高程模型"),
    ("GAMMA/DSM", "伽马软件／数字表面模型"),
    ("GAMMA", "伽马软件"),
    ("Shapefile", "建筑矢量"),
    ("SAR", "合成孔径雷达"),
    ("PSI", "永久散射体干涉测量"),
    ("PS", "永久散射体"),
    ("FID", "建筑编号"),
    ("clean_id", "建筑编号"),
    ("Bottom footprint", "建筑底面轮廓"),
    ("Roof outline", "建筑屋顶轮廓"),
    ("Bottom triangles", "底面三角形"),
    ("Wall triangles", "墙面三角形"),
    ("Roof triangles", "屋顶三角形"),
    ("Bottom", "底面"),
    ("Wall", "墙面"),
    ("Roof", "屋顶"),
    ("Accepted", "已接受"),
    ("Rejected", "已拒绝"),
    ("Iteration", "迭代次数"),
    ("Coherence", "相干性"),
    ("median", "中位数"),
    ("Mean", "平均"),
    ("Error", "误差"),
    ("error", "误差"),
    ("Reference", "参考"),
    ("Initial", "初始"),
    ("Final", "最终"),
    ("Optimized", "优化后"),
    ("Projected", "投影"),
    ("Estimated", "估计"),
    ("Building", "建筑"),
    ("buildings", "栋建筑"),
    ("pixels", "像素"),
    ("pixel", "像素"),
    ("score", "得分"),
    ("quality", "质量"),
    ("height", "高度"),
]


# 对可见图注使用完整单词替换，避免“m”误伤 mask/model/medium。
WORD_REPLACEMENTS = {
    "absolute": "绝对", "accepted": "已接受", "across": "跨", "adaptive": "自适应",
    "after": "之后", "always": "始终", "ambiguity": "模糊性", "ambiguous": "模糊的",
    "amplitude": "幅度", "anchored": "锚定", "and": "与", "applied": "已应用",
    "area": "全区", "as": "作为", "assessment": "评估", "assignment": "归属",
    "assignments": "归属", "azimuth": "方位向", "base": "基底", "baseline": "基线",
    "before": "之前", "bias": "偏差", "bottom": "底面", "boundary": "边界",
    "bounds": "范围", "building": "建筑", "buildings": "建筑", "but": "但",
    "by": "通过", "calibrated": "标定后", "candidate": "候选", "candidates": "候选项",
    "classes": "等级", "classification": "分类", "clipped": "截断", "coherent": "相干",
    "col": "列", "colored": "着色", "combined": "组合", "comparison": "对比",
    "confidence": "置信度", "connectivity": "连通性", "consensus": "共识", "constrained": "约束",
    "convergence": "收敛", "corridor": "廊道", "criterion": "准则", "curves": "曲线",
    "damped": "阻尼", "decrease": "减小", "denotes": "表示", "density": "密度",
    "derived": "反演", "detail": "细节", "diagnostics": "诊断", "difference": "差值",
    "disconnected": "不连通", "display": "显示", "distribution": "分布", "does": "",
    "down": "降低", "easting": "东向坐标", "edge": "边缘", "elevation": "高程",
    "enhanced": "增强后", "enhancement": "增强", "enlargement": "扩展", "equation": "方程",
    "equations": "方程", "estimate": "估计", "estimated": "估计", "estimates": "估计结果",
    "evidence": "证据", "example": "示例", "excluded": "已排除", "feature": "特征",
    "final": "最终", "first": "首次", "footprints": "轮廓", "for": "用于",
    "from": "来自", "full": "全区", "fusion": "融合", "gating": "门控",
    "geometry": "几何", "ground": "地面", "height": "高度", "heights": "高度",
    "high": "高", "identifiability": "可识别性", "identify": "识别", "image": "影像",
    "in": "中", "independent": "独立", "inliers": "内点", "input": "输入",
    "inserted": "已写入", "inside": "内部", "internal": "内部", "invalid": "无效",
    "inversion": "反演", "is": "为", "iterations": "迭代", "kept": "保留",
    "labels": "标注", "level": "等级", "local": "局部", "locator": "定位",
    "low": "低", "mapped": "映射", "mapping": "映射", "mask": "掩膜",
    "max": "最大", "median": "中位数", "medium": "中", "more": "更多",
    "multi": "多景", "no": "无", "northing": "北向坐标", "not": "不",
    "observations": "观测", "of": "的", "offset": "偏移", "on": "于",
    "only": "仅", "optimized": "优化后", "or": "或", "orange": "橙色",
    "other": "其他", "outline": "轮廓", "outlines": "轮廓", "outside": "外部",
    "over": "覆盖", "override": "替代", "partial": "部分", "peak": "峰值",
    "per": "每", "percentile": "百分位", "plan": "平面", "position": "位置",
    "prior": "先验", "projected": "已投影", "projection": "投影", "projections": "投影",
    "quality": "质量", "range": "距离向", "raw": "原始", "recommended": "推荐",
    "reference": "参考", "refined": "精化后", "refinement": "精化", "refinements": "精化结果",
    "registration": "配准", "rejected": "已拒绝", "removed": "已移除", "representative": "代表性",
    "residual": "残差", "resolve": "解决", "restoration": "恢复", "restored": "已恢复",
    "retained": "已保留", "reveal": "显示", "rise": "高层", "robust": "稳健",
    "roof": "屋顶", "roofs": "屋顶", "rooftop": "屋顶", "rounded": "取整",
    "row": "行", "same": "相同", "scene": "场景", "screening": "筛选",
    "search": "搜索", "selection": "选择", "sensitivity": "敏感性", "sharp": "明确",
    "show": "显示", "single": "单景", "solution": "解", "solutions": "解",
    "spatially": "空间标定", "stability": "稳定性", "stabilize": "趋于稳定", "stable": "稳定",
    "standardized": "标准化", "stopping": "停止", "strict": "严格", "strong": "强",
    "support": "支持", "supported": "支持", "surface": "表面", "surfaces": "表面",
    "tail": "尾部", "the": "", "three": "三景", "threshold": "阈值",
    "to": "至", "top": "顶部", "triangle": "三角面", "triangles": "三角面",
    "triangular": "三角面", "truth": "真值", "uncertainty": "不确定性", "unrefined": "未精化",
    "unstable": "不稳定", "unsupported": "无支持", "update": "更新", "updates": "更新量",
    "used": "使用", "using": "使用", "valid": "有效", "validation": "验证",
    "values": "数值", "vector": "矢量", "vertices": "顶点", "view": "视图",
    "wall": "墙面", "weak": "弱", "weighted": "加权", "wide": "宽",
    "width": "宽度", "with": "与",
    "native": "原始", "resampled": "重采样后",
}

ACRONYM_REPLACEMENTS = {
    "RSLC": "雷达复数影像", "SAR": "合成孔径雷达", "PSI": "永久散射体干涉测量",
    "PS": "永久散射体", "LOS": "视线方向", "ECEF": "地心地固坐标", "UTM": "通用横轴墨卡托",
    "WGS84": "1984世界大地坐标系", "DSM": "数字表面模型", "DEM": "数字高程模型",
    "SHP": "建筑矢量", "LUT": "查找表", "MAE": "平均绝对误差", "RMSE": "均方根误差",
    "RMS": "均方根", "FID": "建筑编号", "N/A": "不适用", "Top-K": "前K项",
}


def numeric_id(text: str) -> str:
    return str(int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:5], "big"))


def chinese_stem(stem: str) -> str:
    original = stem
    value = stem.replace("-", "_")
    for english, chinese in sorted(PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(english, chinese)
    token_replacements = {
        "fig": "图",
        "figure": "图",
        "building": "建筑",
        "projection": "投影",
        "comparison": "对比",
        "diagnostic": "诊断",
        "overview": "总览",
        "map": "图",
        "full": "全区",
        "area": "区域",
        "initial": "初始",
        "final": "最终",
        "corrected": "校正后",
        "optimized": "优化后",
        "registered": "配准后",
        "strict": "严格",
        "triangle": "三角面",
        "mask": "掩膜",
        "amplitude": "幅度",
        "resampled": "重采样",
        "preview": "预览",
        "native": "原始网格",
        "style": "风格",
        "quality": "质量",
        "assessment": "评估",
        "validation": "验证",
        "fusion": "融合",
        "scatter": "散射点",
        "points": "点",
        "rooftop": "屋顶",
        "roof": "屋顶",
        "wall": "墙面",
        "height": "高度",
        "process": "过程",
        "audit": "审计",
        "vector": "矢量",
        "local": "局部",
        "same": "同",
        "selected": "选定",
        "radar": "雷达",
        "sar": "合成孔径雷达",
        "ps": "永久散射体",
        "gamma": "伽马软件",
        "dsm": "数字表面模型",
        "dem": "数字高程模型",
        "rpc": "有理多项式系数",
        "utm": "通用横轴墨卡托",
        "vs": "对比",
        "and": "与",
    }
    parts = []
    for token in value.split("_"):
        lower = token.lower()
        if lower in token_replacements:
            parts.append(token_replacements[lower])
        else:
            parts.append(token)
    value = "_".join(parts)
    value = re.sub(r"(?i)v(\d+)", r"第\1版", value)
    value = re.sub(r"(?i)(\d+)p(\d+)", r"\1点\2", value)
    value = re.sub(r"(?i)3d", "三维", value)
    value = re.sub(r"(?i)fid", "建筑编号", value)
    value = re.sub(r"(?i)clean[_ ]?id", "建筑编号", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if re.search(r"[A-Za-z]", value):
        prefix = re.match(r"^\d+", original)
        lead = f"{prefix.group(0)}_" if prefix else ""
        value = f"{lead}图件_{numeric_id(original)}"
    return value or f"图件_{numeric_id(original)}"


def translated_text(text: str) -> str:
    value = text
    repairs = {
        "米ask": "mask", "米apped": "mapped", "米apping": "mapping",
        "米odel": "model", "米edium": "medium", "米etres": "metres",
        "米et": "met", "米argin": "margin", "argi数量": "margin",
        "gai数量": "gain", "高度s": "heights", "屋顶s": "roofs",
    }
    for broken, repaired in repairs.items():
        value = value.replace(broken, repaired)
    for english, chinese in TEXT_REPLACEMENTS:
        value = value.replace(english, chinese)
    # 修复旧版“n=”子串替换对 gain/margin 造成的污染。
    value = value.replace("gai数量", "gain").replace("margi数量", "margin").replace("米margi数量", "margin")
    for english, chinese in sorted(ACRONYM_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(rf"(?<![A-Za-z]){re.escape(english)}(?![A-Za-z])", chinese, value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)(?<![A-Za-z0-9])V(\d+)(?![A-Za-z0-9])", lambda match: f"第{match.group(1)}版", value)
    value = re.sub(r"(?i)(?<![A-Za-z0-9])P(\d{1,2})(?![A-Za-z0-9])", lambda match: f"第{match.group(1)}百分位", value)
    value = re.sub(r"(?i)\b(\d+)(st|nd|rd|th)\b", r"第\1", value)
    value = re.sub(r"(?i)\bpx\b", "像素", value)
    value = re.sub(r"(?i)\bmetres?\b", "米", value)
    value = re.sub(r"(?i)(?<![A-Za-z])m(?![A-Za-z])", "米", value)
    value = re.sub(r"(?i)\bmodel\b", "模型", value)
    value = re.sub(r"(?i)\bmargin\b", "峰值裕量", value)
    value = re.sub(r"(?i)\bgain\b", "增益", value)
    value = re.sub(r"(?i)\bscore\b", "得分", value)
    value = re.sub(r"(?i)\bscores\b", "得分", value)
    value = re.sub(r"(?i)\bcorrelation\b", "相关性", value)
    value = re.sub(r"(?i)\bcorr\b", "相关性", value)
    value = re.sub(r"(?i)\bdifference\b", "差值", value)
    value = re.sub(r"(?i)\bmean\b", "平均值", value)
    value = re.sub(r"(?i)\bstd\b", "标准差", value)
    value = re.sub(r"(?i)\babove\b", "高于", value)
    value = re.sub(r"(?i)\breceive\b", "接收", value)
    value = re.sub(r"(?i)\bsub(?=像素)", "亚", value)
    value = re.sub(r"(?i)\bmet\b", "满足", value)
    value = re.sub(r"(?i)\bfig\.?\s*", "图", value)
    value = re.sub(r"(?i)gc_map2", "地理编码查找表命令", value)
    value = re.sub(r"(?i)sim_(?:sar|合成孔径雷达)", "雷达模拟命令", value)
    for english, chinese in sorted(WORD_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(rf"(?i)(?<![A-Za-z]){re.escape(english)}(?![A-Za-z])", chinese, value)
    # 图版分栏符也使用中文序号；数学变量 H/r/s 保留为公式符号。
    panel_names = {"a": "甲", "b": "乙", "c": "丙", "d": "丁", "e": "戊", "f": "己"}
    value = re.sub(
        r"^\s*([a-f])(?=\s)",
        lambda match: f"（{panel_names[match.group(1).lower()]}）",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\s*\(([a-f])\)(?=\s)", lambda match: f"（{panel_names[match.group(1).lower()]}）", value, flags=re.IGNORECASE)
    value = value.replace("a、b统一色标", "甲、乙统一色标")
    value = value.replace("A/B", "甲／乙").replace("C/D", "丙／丁")
    value = re.sub(r"(?i)TerraSAR-X", "星载合成孔径雷达", value)
    value = value.replace("3-D", "三维").replace("3D", "三维")
    value = value.replace("高度s", "高度").replace("建筑s", "建筑").replace("屋顶s", "屋顶")
    value = value.replace("米mmargin", "峰值裕量").replace("mmargin", "峰值裕量")
    value = value.replace("前K项", "前若干项")
    value = value.replace("至 a 建筑", "至一栋建筑")
    exact_cleanup = {
        "平面 视图 的 永久散射体干涉测量 高度 覆盖 建筑 轮廓": "建筑轮廓上的永久散射体干涉测量高度平面图",
        "底面=4,059, 墙面=12,230, 屋顶=4,059 | 最终 稳定 高度 已写入=307": "底面=4,059，墙面=12,230，屋顶=4,059｜最终写入307个稳定高度",
        "建筑 高度 已保留 之后 每-建筑 收敛 筛选 — 数值 中 米": "逐建筑收敛筛选后保留的建筑高度（单位：米）",
        "建筑 与 已接受 高度": "具有已接受高度的建筑",
    }
    for rough, polished in exact_cleanup.items():
        value = value.replace(rough, polished)
    value = re.sub(r"(?i)\bn\s*=", "数量=", value)
    value = re.sub(r"(?<=\d)N(?=\b|\D)", "北带", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_版本{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def move_collection(source: Path, destination: Path, mappings: list[tuple[Path, Path]]) -> None:
    if not source.exists() or source == destination or source in destination.parents:
        return
    for item in sorted(source.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target = unique_target(target)
        shutil.move(str(item), str(target))
        mappings.append((item, target))
    for directory in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        source.rmdir()
    except OSError:
        pass


def flatten_format_directories(root: Path, mappings: list[tuple[Path, Path]]) -> None:
    if not root.exists():
        return
    for directory in sorted(root.rglob("*"), reverse=True):
        if not directory.is_dir() or directory.name.lower() not in {"png", "svg"}:
            continue
        for item in directory.iterdir():
            if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = unique_target(directory.parent / item.name)
            shutil.move(str(item), str(target))
            mappings.append((item, target))
        try:
            directory.rmdir()
        except OSError:
            pass


def rename_images(root: Path, mappings: list[tuple[Path, Path]]) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        new_name = chinese_stem(path.stem) + path.suffix.lower()
        if new_name == path.name:
            continue
        target = unique_target(path.with_name(new_name))
        path.rename(target)
        mappings.append((path, target))


def translate_svg(path: Path) -> bool:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return False
    changed = False
    for element in tree.iter():
        if element.tag.rsplit("}", 1)[-1] not in {"text", "tspan"}:
            continue
        if element.text:
            translated = translated_text(element.text)
            if translated != element.text:
                element.text = translated
                changed = True
    if changed:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return changed


def rerender_paired_png(svg: Path) -> bool:
    """用 Python/CairoSVG 把同名 PNG 与中文 SVG 保持一致。"""
    png = svg.with_suffix(".png")
    if not png.exists():
        return False
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg), write_to=str(png))
    except Exception as exc:  # 个别 SVG 不应阻断整批整理
        print(f"PNG 重新渲染失败: {svg.relative_to(WORKSPACE)}: {exc}")
        return False
    return True


def write_manifest(mappings: list[tuple[Path, Path]], translated_svgs: list[Path]) -> None:
    output = WORKSPACE / "图片整理映射.md"
    previous = []
    if output.exists():
        previous = [line for line in output.read_text(encoding="utf-8").splitlines() if line.startswith("- `")]
    lines = [
        "# 图片整理映射",
        "",
        "本文件记录图片目录归并和中文文件名映射，用于追溯历史路径。",
        "",
        f"- 本轮路径或文件名变更：{len(mappings)}",
        f"- 中文化 SVG：{len(translated_svgs)}",
        "",
        "## 路径映射",
        "",
    ]
    lines.extend(previous)
    for old, new in mappings:
        lines.append(f"- `{old.relative_to(WORKSPACE)}` → `{new.relative_to(WORKSPACE)}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mappings: list[tuple[Path, Path]] = []
    for source, destination in COLLECTIONS:
        move_collection(source, destination, mappings)
    picall_roots = [
        WORKSPACE / "geocoding/results/picall",
        WORKSPACE / "geocoding/siwei_image1_volume_geocode/results/picall",
        WORKSPACE / "projection_correction/touying/results/picall",
        WORKSPACE / "projection_correction/touying_roof_workflow/results/picall",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/results/picall",
        WORKSPACE / "height_estimation/building_height_estimation_pixel_offset/siwei_gaojing_reproduction/results/picall",
        WORKSPACE / "height_estimation/building_height_estimation_roof_only/results/picall",
        WORKSPACE / "height_estimation/ps_triangle_height_estimation/results/picall",
        WORKSPACE / "output/像素偏移建筑高度估计PICALL结果/picall",
        WORKSPACE / "output/同济校区PS三角面建筑估高方法报告/picall",
    ]
    for root in picall_roots:
        flatten_format_directories(root, mappings)
        rename_images(root, mappings)
    translated_svgs = []
    rerendered_pngs = []
    for root in picall_roots:
        if not root.exists():
            continue
        for svg in root.rglob("*.svg"):
            if translate_svg(svg):
                translated_svgs.append(svg)
            if rerender_paired_png(svg):
                rerendered_pngs.append(svg.with_suffix(".png"))
    write_manifest(mappings, translated_svgs)
    print(f"mappings={len(mappings)}")
    print(f"translated_svgs={len(translated_svgs)}")
    print(f"rerendered_pngs={len(rerendered_pngs)}")


if __name__ == "__main__":
    main()
