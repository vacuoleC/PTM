"""Generate six publication-style Chinese figures for the PTM stage report.

The project environment's Matplotlib renderer exits unexpectedly on this
Windows host.  This script therefore uses Pillow only; all numbers are read
from the versioned CSV results under ``outputs/``.

Run from the repository root:
    D:\enviranment\ptm-encoder\python.exe .doc\scripts\generate_stage_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / ".doc" / "figures"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")

BLUE = "#2A6F97"
ORANGE = "#D97706"
GREEN = "#2F855A"
RED = "#C2410C"
GREY = "#94A3B8"
DARK = "#1F2937"
GRID = "#D9E2EC"
LIGHT = "#E8F0F5"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return a Chinese-capable font bundled with Windows."""
    return ImageFont.truetype(str(FONT_PATH), size=size, index=0)


def canvas(width: int = 3600, height: int = 1500) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a white, 300 dpi canvas suitable for a report or slide."""
    image = Image.new("RGB", (width, height), WHITE)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, filename: str) -> None:
    """Save one PNG using a fixed resolution and the report figure directory."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(FIGURES / filename, dpi=(300, 300))


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.FreeTypeFont, fill: str = DARK) -> None:
    """Draw multi-line text centred at an exact pixel coordinate."""
    draw.multiline_text(xy, text, font=fnt, fill=fill, anchor="mm", align="center", spacing=8)


def title(draw: ImageDraw.ImageDraw, text: str, width: int) -> None:
    """Draw the shared report-figure title treatment."""
    draw.text((115, 70), text, font=font(45), fill=DARK)
    draw.line((115, 145, width - 115, 145), fill=GRID, width=4)


def panel(draw: ImageDraw.ImageDraw, label: str, x: int, y: int) -> None:
    """Draw a conventional panel label."""
    draw.text((x, y), label, font=font(46), fill=DARK)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], outline: str, text: str, fill: str = WHITE) -> None:
    """Draw a workflow or decision box with centred Chinese text."""
    draw.rounded_rectangle(box, radius=26, fill=fill, outline=outline, width=5)
    center(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font(31), DARK)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = GREY) -> None:
    """Draw a simple directed connector with a triangular arrowhead."""
    draw.line((start, end), fill=color, width=6)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = end
    p2 = (end[0] - int(32 * ux) + int(15 * px), end[1] - int(32 * uy) + int(15 * py))
    p3 = (end[0] - int(32 * ux) - int(15 * px), end[1] - int(32 * uy) - int(15 * py))
    draw.polygon((p1, p2, p3), fill=color)


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ylabel: str, xmin: float, xmax: float, ymin: float, ymax: float, ticks: list[float]) -> None:
    """Draw a minimal shared-coordinate plot frame and y-axis grid."""
    left, top, right, bottom = box
    draw.line((left, top, left, bottom, right, bottom), fill=DARK, width=4)
    for tick in ticks:
        y = bottom - (tick - ymin) / (ymax - ymin) * (bottom - top)
        draw.line((left, int(y), right, int(y)), fill=GRID, width=2)
        draw.text((left - 18, int(y)), f"{tick:.1f}", font=font(24), fill=DARK, anchor="rm")
    draw.text((left - 105, (top + bottom) / 2), ylabel, font=font(25), fill=DARK, anchor="mm")


def map_y(value: float, box: tuple[int, int, int, int], ymin: float, ymax: float) -> int:
    """Convert a data y value to the top-origin image coordinate system."""
    _, top, _, bottom = box
    return int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top))


def workflow() -> None:
    """Figure 1: research question, decision gate, and route conclusion."""
    image, draw = canvas(3600, 1500)
    title(draw, "研究路线：从选题到阶段性路线决策", image.width)
    boxes = [
        ((130, 340, 850, 700), BLUE, "选题立意\n定量 PTM 是否需要\n专门的 encoder？"),
        ((1010, 340, 1730, 700), BLUE, "数据基础\nCPTAC 多 PTM 残差\n校正母蛋白丰度"),
        ((1890, 340, 2610, 700), ORANGE, "唯一处决门\nPhase 0：肿瘤/正常\n是否存在可学习信号？"),
        ((2770, 340, 3490, 700), GREEN, "必要性检验\n困难任务、消融、PCA\n与冻结 encoder 对照"),
    ]
    for box, color, text in boxes:
        rounded(draw, box, color, text)
    for start, end in [((850, 520), (1010, 520)), ((1730, 520), (1890, 520)), ((2610, 520), (2770, 520))]:
        arrow(draw, start, end)
    rounded(draw, (1160, 880, 1960, 1240), ORANGE, "qPTM 可行性审计\n位点重叠足够，但配对\n多类条件仅 2 个")
    rounded(draw, (2390, 880, 3290, 1240), RED, "阶段结论\n不扩展当前联合多类\nqPTM+CPTAC encoder")
    arrow(draw, (2310, 700), (1660, 880))
    arrow(draw, (1960, 1060), (2390, 1060))
    draw.text((130, 1325), "原则：Phase 0 通过才允许项目继续；其余阴性结果用于资源路由，不等同于否定 PTM 或否定未来所有 encoder 方案。", font=font(29), fill=DARK)
    save(image, "figure_1_research_workflow.png")


def data_coverage() -> None:
    """Figure 2: multi-PTM cohort coverage and residual-matrix definition."""
    coverage = pd.read_csv(OUTPUTS / "coverage_matrix.csv")
    image, draw = canvas(3300, 1500)
    title(draw, "数据基础：可用队列与生物化学校正", image.width)
    panel(draw, "A", 120, 190); panel(draw, "B", 1810, 190)
    chart = (250, 310, 1550, 1200)
    axes(draw, chart, "三个组学共同可用样本数", 0, 1, 0, 135, [0, 25, 50, 75, 100, 125])
    groups = [570, 900, 1230]
    for x, (_, row) in zip(groups, coverage.iterrows()):
        tumor_top = map_y(float(row["common_tumor"]), chart, 0, 135)
        normal_top = map_y(float(row["common_normal"]), chart, 0, 135)
        draw.rectangle((x - 110, tumor_top, x - 15, 1200), fill=BLUE)
        draw.rectangle((x + 15, normal_top, x + 110, 1200), fill=GREEN)
        center(draw, (x - 62, tumor_top - 35), str(int(row["common_tumor"])), font(24))
        center(draw, (x + 62, normal_top - 35), str(int(row["common_normal"])), font(24))
        center(draw, (x, 1260), row["cancer"], font(27))
    threshold = map_y(20, chart, 0, 135)
    draw.line((250, threshold, 1550, threshold), fill=DARK, width=3)
    draw.text((1280, threshold - 37), "正常样本功效下限 = 20", font=font(22), fill=DARK)
    draw.rectangle((350, 220, 380, 250), fill=BLUE); draw.text((392, 218), "肿瘤样本", font=font(22), fill=DARK)
    draw.rectangle((600, 220, 630, 250), fill=GREEN); draw.text((642, 218), "配对正常样本", font=font(22), fill=DARK)

    draw.text((1840, 315), "分析单位：经母蛋白校正的 PTM 残差", font=font(35), fill=DARK)
    rows = [("输入层", "磷酸化 + 乙酰化 + 蛋白组"), ("校正", "逐位点：PTM ~ 母蛋白；取残差"), ("目的", "减少蛋白总量变化造成的混淆"), ("模型输入", "多 PTM 残差矩阵，而非原始强度")]
    for i, (key, value) in enumerate(rows):
        y = 470 + i * 165
        draw.text((1870, y), key, font=font(30), fill=BLUE)
        draw.text((2180, y), value, font=font(30), fill=DARK)
        if i < len(rows) - 1:
            draw.line((1870, y + 90, 3150, y + 90), fill=GRID, width=3)
    draw.multiline_text((1870, 1190), "这一步不是估计绝对修饰占据率，\n而是使比较更少受母蛋白丰度影响。", font=font(28), fill=DARK, spacing=10)
    save(image, "figure_2_data_coverage.png")


def experimental_evidence() -> None:
    """Figure 3: Phase 0, repeated multi-PTM increment, and encoder comparison."""
    phase0 = pd.read_csv(OUTPUTS / "phase0_xgboost_null_summary.csv")
    repeated = pd.read_csv(OUTPUTS / "lscc_grade_g2_vs_g3_repeated_xgboost_scores.csv")
    increment = pd.read_csv(OUTPUTS / "lscc_grade_g2_vs_g3_increment_summary.csv").iloc[0]
    encoder = pd.read_csv(OUTPUTS / "lscc_grade_g2_vs_g3_encoder_necessity_summary.csv")
    models = encoder.loc[encoder["row_type"] == "model_summary"].reset_index(drop=True)
    image, draw = canvas(4500, 1600)
    title(draw, "核心实验结果：信号存在，但当前 encoder 路线没有可检测增益", image.width)
    charts = [(230, 360, 1400, 1300), (1750, 360, 2920, 1300), (3270, 360, 4440, 1300)]
    for label, x in zip(["A", "B", "C"], [100, 1620, 3140]): panel(draw, label, x, 190)

    box = charts[0]; axes(draw, box, "平均精确率（AUPRC）", 0, 1, 0.4, 1.08, [0.4, 0.6, 0.8, 1.0])
    draw.text((box[0], 270), "Phase 0：肿瘤/正常信号存在", font=font(31), fill=DARK)
    for x, (_, row) in zip([470, 810, 1150], phase0.iterrows()):
        null_y = map_y(float(row["null_average_precision_mean"]), box, 0.4, 1.08); obs_y = map_y(float(row["observed_average_precision"]), box, 0.4, 1.08)
        draw.rectangle((x - 110, null_y, x - 20, box[3]), fill=GREY); draw.rectangle((x + 20, obs_y, x + 110, box[3]), fill=BLUE)
        center(draw, (x + 65, obs_y - 35), f"{row['observed_average_precision']:.3f}", font(23)); center(draw, (x, 1360), row["cohort"], font(26))
        center(draw, (x, 1265), f"q={row['bh_q']:.4f}", font(20))
    draw.rectangle((300, 300, 330, 330), fill=GREY); draw.text((342, 297), "置换 null 均值", font=font(21), fill=DARK)
    draw.rectangle((650, 300, 680, 330), fill=BLUE); draw.text((692, 297), "观察到的 AUPRC", font=font(21), fill=DARK)

    box = charts[1]; axes(draw, box, "重复 CV 的 AUPRC 差", 0, 1, -0.28, 0.28, [-0.2, 0.0, 0.2])
    draw.text((box[0], 270), "困难任务：乙酰化无稳定增量", font=font(31), fill=DARK)
    pivot = repeated.pivot(index=["repeat", "fold"], columns="feature_set", values="average_precision")
    deltas = (pivot["multi_ptm"] - pivot["phosphoproteome"]).dropna().to_numpy()
    rng = np.random.default_rng(0)
    for value in deltas:
        x = int(2320 + rng.uniform(-135, 135)); y = map_y(float(value), box, -0.28, 0.28)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=ORANGE)
    zero = map_y(0, box, -0.28, 0.28); draw.line((box[0], zero, box[2], zero), fill=DARK, width=3)
    q1, med, q3 = np.quantile(deltas, [0.25, 0.5, 0.75]); low, high = np.min(deltas), np.max(deltas)
    xmid = 2320
    draw.line((xmid, map_y(low, box, -0.28, 0.28), xmid, map_y(high, box, -0.28, 0.28)), fill=ORANGE, width=5)
    draw.rectangle((xmid - 65, map_y(q3, box, -0.28, 0.28), xmid + 65, map_y(q1, box, -0.28, 0.28)), fill=LIGHT, outline=ORANGE, width=4)
    draw.line((xmid - 65, map_y(med, box, -0.28, 0.28), xmid + 65, map_y(med, box, -0.28, 0.28)), fill=ORANGE, width=5)
    center(draw, (2320, 1380), "多类 PTM − 磷酸化", font(25))
    rounded(draw, (1950, 420, 2700, 570), GRID, f"均值 = {increment['repeated_delta_mean']:+.4f}\n校正单侧 p = {increment['nadeau_bengio_p_one_sided']:.3f}")

    box = charts[2]; axes(draw, box, "AUPRC（50 个重复折）", 0, 1, 0.35, 0.65, [0.4, 0.5, 0.6])
    draw.text((box[0], 270), "冻结 encoder 未优于公平 PCA", font=font(31), fill=DARK)
    names = ["原始共同特征\n逻辑回归", "PCA\n64 维", "冻结 encoder\n64 维"]
    for x, (_, row), color, name in zip([3500, 3850, 4200], models.iterrows(), [GREY, ORANGE, GREEN], names):
        value = float(row["average_precision_mean"]); std = float(row["average_precision_std"]); top = map_y(value, box, 0.35, 0.65)
        draw.rectangle((x - 95, top, x + 95, box[3]), fill=color)
        lo, hi = map_y(value - std, box, 0.35, 0.65), map_y(value + std, box, 0.35, 0.65)
        draw.line((x, lo, x, hi), fill=DARK, width=4); draw.line((x - 18, lo, x + 18, lo), fill=DARK, width=4); draw.line((x - 18, hi, x + 18, hi), fill=DARK, width=4)
        center(draw, (x, top - 35), f"{value:.3f}", font(22)); center(draw, (x, 1405), name, font(21))
    chance = map_y(48 / 106, box, 0.35, 0.65); draw.line((box[0], chance, box[2], chance), fill=DARK, width=3)
    draw.text((3860, chance + 15), "G3 比例", font=font(20), fill=DARK)
    center(draw, (3850, 470), "encoder − PCA = +0.0012\n校正单侧 p = 0.491", font(24))
    save(image, "figure_3_experimental_evidence.png")


def training_curve() -> None:
    """Figure 4: self-supervised reconstruction loss over 150 epochs."""
    history = pd.read_csv(OUTPUTS / "lscc_grade_g2_vs_g3_encoder_training_history.csv")
    image, draw = canvas(2700, 1500)
    title(draw, "CPTAC-only 冻结 encoder 的无标签预训练已收敛", image.width)
    panel(draw, "A", 120, 190)
    box = (350, 340, 2450, 1240); axes(draw, box, "遮蔽重建均方误差", 0, 150, 0.5, 1.1, [0.6, 0.8, 1.0])
    pts = []
    for _, row in history.iterrows():
        x = int(box[0] + (float(row["epoch"]) - 1) / 149 * (box[2] - box[0])); y = map_y(float(row["masked_mse"]), box, 0.5, 1.1); pts.append((x, y))
    draw.line(pts, fill=GREEN, width=6)
    for epoch in [1, 150]:
        row = history.loc[history["epoch"] == epoch].iloc[0]; x, y = pts[epoch - 1]
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=GREEN)
        draw.text((x + (20 if epoch == 1 else -280), y - 65), f"第 {epoch} 轮：{row['masked_mse']:.3f}", font=font(26), fill=DARK)
    for epoch in [1, 30, 60, 90, 120, 150]:
        x = int(box[0] + (epoch - 1) / 149 * (box[2] - box[0])); center(draw, (x, 1300), str(epoch), font(24))
    center(draw, ((box[0] + box[2]) / 2, 1380), "预训练轮次", font(28))
    rounded(draw, (1780, 420, 2350, 650), GRID, "577 个无标签样本\n43,647 个共同特征\n64 维潜变量")
    save(image, "figure_4_encoder_pretraining_curve.png")


def qptm_feasibility() -> None:
    """Figure 5: qPTM condition coverage, site overlap, and false-pairing risk."""
    condition = pd.read_csv(OUTPUTS / "qptm_condition_audit.csv")
    overlap = pd.read_csv(OUTPUTS / "qptm_cptac_feature_overlap_audit.csv")
    image, draw = canvas(4500, 1600)
    title(draw, "qPTM 审计：可迁移位点不等于存在真实配对多类条件", image.width)
    for label, x in zip(["A", "B", "C"], [100, 1620, 3140]): panel(draw, label, x, 190)
    box = (230, 360, 1400, 1300); axes(draw, box, "qPTM 条件上下文数（对数尺度）", 0, 1, 0, 1, [])
    draw.text((box[0], 270), "单类数据丰富，配对多类数据稀缺", font=font(31), fill=DARK)
    values = [1167, 82, 2]; labels = ["磷酸化\n条件", "乙酰化\n条件", "真实配对\n多类条件"]
    def logy(v: float) -> int: return int(box[3] - (np.log10(v) / np.log10(2000)) * (box[3] - box[1]))
    for tick in [1, 10, 100, 1000]:
        y = logy(tick); draw.line((box[0], y, box[2], y), fill=GRID, width=2); draw.text((box[0] - 20, y), str(tick), font=font(23), fill=DARK, anchor="rm")
    for x, value, color, label in zip([470, 810, 1150], values, [BLUE, ORANGE, RED], labels):
        top = logy(value); draw.rectangle((x - 90, top, x + 90, box[3]), fill=color); center(draw, (x, top - 38), f"{value:,}", font(25)); center(draw, (x, 1400), label, font(23))
    box = (1750, 360, 2920, 1300); axes(draw, box, "位点匹配比例（%）", 0, 1, 0, 100, [0, 25, 50, 75, 100])
    draw.text((box[0], 270), "位点空间可以对齐", font=font(31), fill=DARK)
    for x, (_, row), label in zip([2070, 2570], overlap.iterrows(), ["磷酸化", "乙酰化"]):
        for offset, value, color in [(-72, row["cptac_feature_match_fraction"] * 100, GREEN), (72, row["qptm_site_match_fraction"] * 100, GREY)]:
            top = map_y(float(value), box, 0, 100); draw.rectangle((x + offset - 55, top, x + offset + 55, box[3]), fill=color); center(draw, (x + offset, top - 35), f"{value:.1f}%", font(22))
        center(draw, (x, 1380), label, font(25))
    draw.rectangle((1800, 305, 1830, 335), fill=GREEN); draw.text((1840, 300), "CPTAC 特征可匹配", font=font(20), fill=DARK)
    draw.rectangle((2240, 305, 2270, 335), fill=GREY); draw.text((2280, 300), "qPTM 位点可匹配", font=font(20), fill=DARK)
    draw.text((3270, 270), "为何不能拼接成联合训练矩阵", font=font(31), fill=DARK)
    rounded(draw, (3260, 520, 3710, 810), BLUE, "研究 A\n仅测磷酸化\n条件 α")
    rounded(draw, (3260, 970, 3710, 1260), ORANGE, "研究 B\n仅测乙酰化\n条件 β")
    rounded(draw, (3930, 735, 4420, 1035), RED, "若按行强行拼接\n会虚构 α 与 β 的\n共变关系")
    arrow(draw, (3710, 665), (3930, 805)); arrow(draw, (3710, 1115), (3930, 965))
    draw.multiline_text((3260, 1330), "正确结论：保留单类 qPTM 的价值，\n但停止当前“联合配对多类”方案。", font=font(26), fill=DARK, spacing=8)
    save(image, "figure_5_qptm_feasibility.png")


def decision_matrix() -> None:
    """Figure 6: distinguish project feasibility from the current route decision."""
    image, draw = canvas(3600, 1600)
    title(draw, "证据到决策：通过的是项目可行性，不是当前 encoder 路线", image.width)
    columns = [150, 900, 1640, 2460]
    for x, head in zip(columns, ["证据模块", "关键结果", "能回答什么", "对路线的含义"]): draw.text((x, 250), head, font=font(31), fill=DARK)
    rows = [("Phase 0", "三队列 q=0.0099", "PTM 是否有疾病信号", "通过唯一 no-go 门", GREEN), ("困难任务消融", "ΔAUPRC = -0.0060", "乙酰化是否稳定增量", "当前任务：不支持", ORANGE), ("冻结 encoder", "encoder−PCA = +0.0012", "压缩表征是否更优", "当前模型：不支持", ORANGE), ("qPTM 审计", "真实配对多类条件 = 2", "能否联合条件级预训练", "停止伪配对方案", RED)]
    for i, (module, result, question, implication, color) in enumerate(rows):
        y = 360 + i * 230
        draw.rounded_rectangle((100, y, 3500, y + 165), radius=18, fill=WHITE, outline=GRID, width=3)
        draw.rounded_rectangle((115, y + 18, 145, y + 147), radius=8, fill=color)
        for x, text in zip(columns, [module, result, question, implication]): draw.text((x, y + 57), text, font=font(29), fill=DARK)
    draw.text((150, 1425), "最终路由：保留 CPTAC encoder 管线作为未来基线；下一优先级是寻找 PTM-only 锚点或真实配对多类 PTM 数据。", font=font(29), fill=DARK)
    save(image, "figure_6_evidence_to_decision.png")


def main() -> None:
    """Generate every image used by the stage report."""
    workflow(); data_coverage(); experimental_evidence(); training_curve(); qptm_feasibility(); decision_matrix()
    print(f"Saved six figures to {FIGURES}")


if __name__ == "__main__":
    main()
