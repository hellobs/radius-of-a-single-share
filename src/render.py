# -*- coding: utf-8 -*-
"""
《一次转发的半径》—— 径向生成艺术渲染
======================================
读取 Layer 1 仿真事件表，把"每一次转发"画成从源头向外生长的一圈枝蔓：

  A 联（无辟谣）：谣言传播树 —— 中心炽热金黄（首发）→ 边缘深红（失控蔓延）
  B 联（辟谣介入）：被截断的谣言残枝（暖色细线）+ 真相之树
    （清朗蓝；六个官方信源在 t=activate_t 环绕圆心同时上线，真相向外漫开）

画面不含任何文字；标题、数据注脚由 compose.py 排版叠加。
径向布局：BFS 深度 = 半径（环），角度按子树规模分配（sunburst 式），
转发边画成向心收拢的贝塞尔弧线，三层叠加（光晕/中间/核心）模拟发光。

用法
  python src/render.py --preview   # 150dpi 快速预览 → output/preview_dual.png
  python src/render.py             # 300dpi 成片画布 → output/render_dual.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import DATA_DIR, ROOT, load_config  # noqa: E402

OUT = ROOT / "output"

# ---- 视觉语言（与 config.poster 色彩语义一致）----
CMAP_RUMOR = LinearSegmentedColormap.from_list(
    "rumor", [(0.0, "#FFC94D"), (0.35, "#F6A21C"), (0.70, "#E4571F"), (1.0, "#7A1204")])
CMAP_TRUTH = LinearSegmentedColormap.from_list(
    "truth", ["#0E3A66", "#1B6BB0", "#4FA3DC", "#B4E2F8"])   # 深蓝 → 清朗
SEED_COLOR = "#FFF3D6"       # 谣言源头
DEBUNK_COLOR = "#F5C542"     # 官方辟谣号标记（金环）
GUIDE_COLOR = "#8A7A66"      # 深度参考环

# ---- 画布布局（相对画布宽/高的比例；文字区留给 compose.py）----
# matplotlib 坐标系 y 向上；顶部标题区约占 0.17H，底部署名区约占 0.08H
PANEL_A_CENTER = (0.36, 0.653)  # A 联圆心（从顶往下 0.347H）
PANEL_B_CENTER = (0.64, 0.270)  # B 联圆心（从顶往下 0.730H）
PANEL_R = 0.25                  # 面板半径 / 画布宽


def load_group(group):
    ev = json.loads((DATA_DIR / f"{group}_events.json").read_text(encoding="utf-8"))
    stats = json.loads((DATA_DIR / f"{group}_stats.json").read_text(encoding="utf-8"))
    rumor = [(int(t), int(p), int(c)) for t, p, c in ev["rumor_events"]]
    truth = [(int(t), int(p), int(c)) for t, p, c in ev["truth_events"]]
    return rumor, truth, stats


def layout_tree(events, root):
    """时间序事件表 → 径向树布局（深度=环，角度按子树规模分配）。"""
    parent = {root: None}
    children = {}
    depth = {root: 0}
    tnode = {root: 0}
    for t, p, c in events:                      # 事件表按时间序，父必先出现
        parent[c] = p
        depth[c] = depth[p] + 1
        tnode[c] = t
        children.setdefault(p, []).append(c)

    order = list(depth)
    size = {u: 1 for u in order}
    for u in reversed(order):                   # 逆插入序：子先于父被累加
        pu = parent[u]
        if pu is not None:
            size[pu] += size[u]

    span = {root: (0.0, 2.0 * np.pi)}
    angle = {root: 0.0}
    for u in order:
        kids = children.get(u)
        if not kids:
            continue
        s0, s1 = span[u]
        total = size[u] - 1
        cur = s0
        for c in kids:
            w = (s1 - s0) * size[c] / total
            span[c] = (cur, cur + w)
            angle[c] = cur + w / 2.0
            cur += w

    edges = [(depth[parent[u]], angle[parent[u]], depth[u], angle[u], tnode[u])
             for u in order if parent[u] is not None]
    leaves = [u for u in order if u not in children]
    return {"edges": edges, "angle": angle, "depth": depth, "tnode": tnode,
            "leaves": leaves, "max_depth": max(depth.values()), "root": root}


def edge_segments(edges, cx, cy, ring, n=22):
    """转发边 → 极坐标插值弧线：半径线性、角度余弦缓动（不内凹、无回勾）。"""
    segs = np.empty((len(edges), n, 2))
    ts = np.linspace(0.0, 1.0, n)
    ease = (1.0 - np.cos(np.pi * ts)) / 2.0
    for i, (d1, th1, d2, th2, _) in enumerate(edges):
        r = d1 * ring + (d2 - d1) * ring * ts
        dth = (th2 - th1 + np.pi) % (2.0 * np.pi) - np.pi
        th = th1 + dth * ease
        segs[i, :, 0] = cx + r * np.cos(th)
        segs[i, :, 1] = cy + r * np.sin(th)
    return segs


def draw_tree(ax, tree, cx, cy, ring, cmap, lw=0.75,
              passes=((7.0, 0.06), (3.0, 0.22), (1.0, 0.90))):
    """按深度着色（中心炽热/深 → 边缘深红/清朗）；线宽随深度渐细（干粗梢细）；
    三层叠加（光晕/中间/核心）+ 叶尖微粒。"""
    edges = tree["edges"]
    if not edges:
        return
    dmax = tree["max_depth"]
    dd = np.array([e[3] for e in edges], float)                  # 子节点深度
    cols = cmap(dd / dmax)
    taper = 1.2 - 0.75 * (dd / dmax)                             # 干粗梢细
    segs = edge_segments(edges, cx, cy, ring)
    for scale, alpha in passes:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw * taper * scale,
                                         alpha=alpha, capstyle="round", zorder=3))
    lx = cx + np.array([tree["depth"][u] * ring * np.cos(tree["angle"][u])
                        for u in tree["leaves"]])
    ly = cy + np.array([tree["depth"][u] * ring * np.sin(tree["angle"][u])
                        for u in tree["leaves"]])
    ld = np.array([tree["depth"][u] for u in tree["leaves"]], float)
    ax.scatter(lx, ly, s=11, c=cmap(ld / dmax), alpha=0.9, linewidths=0, zorder=4)


def draw_guides(ax, cx, cy, ring, max_d):
    """极淡的同心深度参考环。"""
    th = np.linspace(0, 2 * np.pi, 256)
    for k in range(1, max_d + 1):
        ax.plot(cx + k * ring * np.cos(th), cy + k * ring * np.sin(th),
                color=GUIDE_COLOR, alpha=0.09, lw=0.7, zorder=1)


def draw_seed(ax, cx, cy, color=None):
    color = color or SEED_COLOR
    ax.scatter([cx], [cy], s=2600, color=color, alpha=0.18, linewidths=0, zorder=5)  # 核心光斑
    ax.scatter([cx], [cy], s=150, color=color, alpha=0.95, linewidths=0, zorder=6)
    ax.scatter([cx], [cy], s=560, facecolors="none", edgecolors=color,
               alpha=0.5, linewidths=1.2, zorder=6)


def draw_debunk(ax, tree, cx, cy, ring, roots):
    for u in roots:
        x = cx + tree["depth"][u] * ring * np.cos(tree["angle"][u])
        y = cy + tree["depth"][u] * ring * np.sin(tree["angle"][u])
        ax.scatter([x], [y], s=280, facecolors="none", edgecolors=DEBUNK_COLOR,
                   linewidths=1.6, alpha=0.95, zorder=6)


def render(out_path, dpi):
    cfg = load_config()
    W, H = cfg["poster"]["width_px"], cfg["poster"]["height_px"]

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(cfg["poster"].get("bg", "#FAF6EF"))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    rumorA, _, statsA = load_group("A")
    rumorB, truthB, statsB = load_group("B")
    R = W * PANEL_R

    # ---- A 联：谣言传播树（无辟谣）----
    axc, ayc = W * PANEL_A_CENTER[0], H * PANEL_A_CENTER[1]
    treeA = layout_tree(rumorA, statsA["seed_node"])
    ringA = R / treeA["max_depth"]
    draw_guides(ax, axc, ayc, ringA, treeA["max_depth"])
    draw_tree(ax, treeA, axc, ayc, ringA, CMAP_RUMOR)
    draw_seed(ax, axc, ayc)

    # ---- B 联：谣言残枝 + 真相之树（辟谣介入）----
    bxc, byc = W * PANEL_B_CENTER[0], H * PANEL_B_CENTER[1]
    treeB_r = layout_tree(rumorB, statsB["seed_node"])
    t_act = statsB["activate_t"]
    VIRT = -1
    truth_synth = [(t_act, VIRT, r) for r in statsB["debunk_nodes"]] + truthB
    treeB_t = layout_tree(truth_synth, VIRT)      # 虚根居中，六个官方信源为第一环
    ringB = R / max(treeB_r["max_depth"], treeB_t["max_depth"])
    draw_guides(ax, bxc, byc, ringB, max(treeB_r["max_depth"], treeB_t["max_depth"]))
    draw_tree(ax, treeB_r, bxc, byc, ringB, CMAP_RUMOR, lw=0.42,
              passes=((4.0, 0.05), (1.8, 0.18), (0.8, 0.42)))   # 残枝：细、暖、退后
    draw_tree(ax, treeB_t, bxc, byc, ringB, CMAP_TRUTH, lw=0.65)
    draw_seed(ax, bxc, byc)
    draw_debunk(ax, treeB_t, bxc, byc, ringB, statsB["debunk_nodes"])

    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"✔ 已渲染 -> {out_path}（{W}x{H}px @ {dpi}dpi）")
    print(f"  A 联: 谣言树 {len(treeA['edges'])} 边 / {treeA['max_depth']} 环 / {len(treeA['leaves'])} 叶尖")
    print(f"  B 联: 谣言残枝 {len(treeB_r['edges'])} 边 / {treeB_r['max_depth']} 环"
          f" | 真相树 {len(treeB_t['edges'])} 边 / {treeB_t['max_depth']} 环")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="150dpi 快速预览")
    ap.add_argument("--out", default=None, help="输出路径覆盖")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.preview:
        render(args.out or (OUT / "preview_dual.png"), dpi=150)
    else:
        render(args.out or (OUT / "render_dual.png"), dpi=300)


if __name__ == "__main__":
    main()
