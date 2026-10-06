# -*- coding: utf-8 -*-
"""
《一次转发的半径》—— 海报排版合成
==================================
在 render.py 的生成艺术画布上叠加文字层（全部版位为画布比例，数据动态读取）：

  顶部   主标题《一次转发的半径》+ 副标语
  右中   A 联注脚（失控的传播 · 无辟谣对照组 + 数据行）
  左中   B 联注脚（被拦住的半径 · 辟谣介入对照组 + 数据行）

字体：思源黑体 CN（SIL OFL 免费商用），assets/fonts/（不入库，README 有来源）
输出 release/poster_一次转发的半径.png（3508x4961 @300dpi, RGB）

用法
  python src/compose.py            # 合成海报
  python src/compose.py --preview  # 另存 1000px 宽预览 output/poster_preview.png
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import DATA_DIR, ROOT, load_config  # noqa: E402

OUT = ROOT / "output"
RELEASE = ROOT / "release"
FONTS = ROOT / "assets" / "fonts"

# ---- 色彩（与 render.py 视觉语言一致）----
INK = (43, 33, 24)         # 近黑暖墨：主标题
INK_SOFT = (74, 59, 48)    # 次级文字
MUTED = (138, 122, 106)    # 眉标 / 来源说明
ORANGE = (228, 87, 31)     # A 联强调
BLUE = (27, 107, 176)      # B 联强调


def font(weight, size):
    path = FONTS / f"SourceHanSansCN-{weight}.otf"
    if not path.exists():
        raise FileNotFoundError(f"缺少字体 {path}，请见 README 下载到 assets/fonts/")
    return ImageFont.truetype(str(path), size)


def tracked_width(draw, text, f, tr):
    return sum(draw.textlength(ch, font=f) for ch in text) + tr * max(0, len(text) - 1)


def draw_tracked(draw, cx, cy, text, f, tr, fill):
    """水平居中 + 字距 tracking 的文字。"""
    total = tracked_width(draw, text, f, tr)
    x = cx - total / 2
    for ch in text:
        draw.text((x, cy), ch, font=f, fill=fill, anchor="lm")
        x += draw.textlength(ch, font=f) + tr


def draw_left(draw, x, cy, text, f, fill, bullet=None, bullet_color=None):
    """左对齐一行文字；bullet 为 'square' 时先画小方块。"""
    if bullet == "square":
        draw.rectangle([x, cy - 9, x + 18, cy + 9], fill=bullet_color or fill)
        x += 42
    draw.text((x, cy), text, font=f, fill=fill, anchor="lm")


def caption_block(draw, x, y_center, accent, title, subtitle, rows):
    """侧栏注脚：强调短线 + 标题 + 副题 + 数据行（垂直居中于 y_center）。"""
    f_title = font("Bold", 92)
    f_sub = font("Medium", 48)
    f_row = font("Regular", 52)
    h = 40 + 130 + 84 + len(rows) * 88
    y = y_center - h / 2
    draw.rectangle([x, y + 14, x + 72, y + 22], fill=accent)          # 强调短线
    draw.text((x, y + 40 + 65), title, font=f_title, fill=INK, anchor="lm")
    draw.text((x, y + 170 + 42), subtitle, font=f_sub, fill=MUTED, anchor="lm")
    yy = y + 254 + 44
    for row in rows:
        draw_left(draw, x, yy, row, f_row, INK_SOFT,
                  bullet="square", bullet_color=accent)
        yy += 88


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    W, H = cfg["poster"]["width_px"], cfg["poster"]["height_px"]
    n = cfg["network"]["n"]

    statsA = json.loads((DATA_DIR / "A_stats.json").read_text(encoding="utf-8"))
    statsB = json.loads((DATA_DIR / "B_stats.json").read_text(encoding="utf-8"))

    def pct(s):
        return f"{s['attack_rate'] * 100:.1f}%"

    img = Image.open(OUT / "render_dual.png").convert("RGB")
    draw = ImageDraw.Draw(img)

    # ---- 顶部文字区 ----
    draw_tracked(draw, W / 2, 470, "一次转发的半径", font("Heavy", 224), 16, INK)
    draw_tracked(draw, W / 2, 740, "转发即扩散，理性是防火墙",
                 font("Medium", 84), 22, INK_SOFT)

    # ---- A 联注脚（右侧，垂直居中于 A 圆心）----
    caption_block(
        draw, x=2340, y_center=H * 0.347, accent=ORANGE,
        title="失控的传播", subtitle="无辟谣对照组",
        rows=[f"感染率 {pct(statsA)}（{statsA['ever_heard_rumor']} / {n} 节点）",
              f"传播半径 {statsA['rumor_radius']} 环",
              f"峰值同时传播 {statsA['peak_spreaders']} 人（t={statsA['peak_t']}）"])

    # ---- B 联注脚（左侧，垂直居中于 B 圆心；靠近右侧的蓝图）----
    caption_block(
        draw, x=430, y_center=H * 0.730, accent=BLUE,
        title="被拦住的半径", subtitle=f"辟谣介入对照组 · t={statsB['activate_t']}",
        rows=[f"感染率 {pct(statsB)}（{statsB['ever_heard_rumor']} / {n} 节点）",
              f"传播半径 {statsB['rumor_radius']} 环",
              f"真相触达 {statsB['truth_reached_final']} 人"])

    RELEASE.mkdir(exist_ok=True)
    out = RELEASE / "poster_一次转发的半径.png"
    img.save(out, optimize=True)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"✔ 海报已合成 -> {out}（{size_mb:.1f} MB）")

    if args.preview:
        pv = img.copy()
        pv.thumbnail((1000, 10000))
        pv.save(OUT / "poster_preview.png")
        print(f"✔ 预览 -> {OUT / 'poster_preview.png'}")

    # 简介字数自检（交稿要求 ≤100 字；与 release/ 简介文档保持一致）
    abstract = ("作品由真实仿真数据渲染：24个本地大模型人格化智能体两轮决策实验校准传播参数，"
                "注入3000节点BA无标度网络谣言-真相竞争模型；辟谣使感染率77.9%降至18.7%。"
                "转发即扩散，理性是防火墙。")
    print(f"ℹ 简介字数自检：{len(abstract)} 字（要求 ≤100）")


if __name__ == "__main__":
    main()
