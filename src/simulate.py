# -*- coding: utf-8 -*-
"""
《一次转发的半径》—— 谣言传播仿真
==================================
BA 无标度社交网络上的谣言/真相竞争传播（MT 谣言模型变体）。

状态编码
  0 IGNORANT    易感者（未知情）
  1 R_SPREADER  谣言传播者（活跃）
  2 R_STIFLER   谣言沉默者（听过谣言后自然停止）
  3 T_IMMUNE    真相免疫者（获知真相，从未传播谣言）
  4 T_SPREADER  真相传播者（官方辟谣号 / 被真相转化并转发的节点）

对照实验
  A 组：无辟谣，纯谣言传播
  B 组：同一网络、同一谣言随机流；activate_t 步后官方辟谣号开始传播真相
  随机流隔离（谣言/真相/网络三个独立 RNG）保证 A/B 介入前轨迹逐事件一致。

用法
  python src/simulate.py [config 路径，默认 src/config.yaml]

输出（data/）
  network_meta.json          网络元信息
  A_events.json / B_events.json   传播事件表 [(t, 父节点, 子节点), ...]
  A_stats.json  / B_stats.json    统计指标与逐步状态计数
"""
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

IGNORANT, R_SPREADER, R_STIFLER, T_IMMUNE, T_SPREADER = 0, 1, 2, 3, 4


def load_config(path=None):
    path = Path(path) if path is not None else ROOT / "src" / "config.yaml"
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def pick_seed_node(G, strategy):
    """谣言源头：'max_degree' 选度最高节点（大V首发）；
    整数则选度最接近该值的节点（普通用户首发——任何人的一次转发）。"""
    if strategy == "max_degree":
        return max(G.nodes, key=lambda u: G.degree(u))
    if isinstance(strategy, int):
        return min(G.nodes, key=lambda u: abs(G.degree(u) - strategy))
    raise ValueError(f"未知 seed_node 策略: {strategy}")


def pick_debunk_nodes(G, exclude, count):
    """官方辟谣号：排除源头后度最高的若干枢纽节点。"""
    ranked = sorted(G.nodes, key=lambda u: G.degree(u), reverse=True)
    return [u for u in ranked if u != exclude][:count]


def build_node_behavior(G, cfg):
    """Layer 2 → Layer 1：把 LLM 校准实验测得的素养分型转发率注入节点。

    协议（写进作品说明，保证可交代）：
      - LLM 实验（data/llm_calibration.json）提供"异质性"——
        不同媒介素养人群的转发倾向差异；
      - 网络动力学整体节奏（平均传播率 λ）仍由 config 控制——
        以总量守恒缩放把分型转发率映射到 3000 个节点；
      - 源头节点视为已信谣者，强制归入低素养分型；
      - 高素养转发率实验值为 0 时取下限 FLOOR，避免节点完全沉默。
    """
    cal_path = DATA_DIR / "llm_calibration.json"
    if not cal_path.exists():
        return None
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    a1 = cal["aggregates"]["round1"]
    floor = 0.05
    hi = max(a1["forward_rate_literate"], floor)
    lo = max(a1["forward_rate_illiterate"], floor)

    n = G.number_of_nodes()
    rng = np.random.default_rng(cfg["reproducibility"].get("type_seed", 88))
    literate = rng.random(n) < 0.5           # 高素养占比 50%，与实验人设均衡一致
    src = pick_seed_node(G, cfg["seed_node"]["strategy"])
    literate[src] = False
    raw = np.where(literate, hi, lo)
    scale = cfg["model"]["lambda"] / raw.mean()
    return {
        "literate": literate,
        "lambda_node": raw * scale,
        "meta": {
            "source": "data/llm_calibration.json",
            "llm_model": cal.get("model"),
            "forward_rate_literate_raw": a1["forward_rate_literate"],
            "forward_rate_illiterate_raw": a1["forward_rate_illiterate"],
            "scale": round(float(scale), 4),
            "seed_forced_illiterate": True,
        },
    }


def tree_depth(children, root):
    """传播树最大深度 = 一次转发的最大半径（环数）。"""
    frontier, seen, depth = [root], {root}, 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if frontier:
            depth += 1
    return depth


def simulate(G, cfg, group, behavior=None):
    """单组仿真。group='A' 纯谣言；group='B' 辟谣介入。"""
    mdl, deb = cfg["model"], cfg["debunk"]
    lam, alpha = mdl["lambda"], mdl["alpha"]
    delta, gamma = mdl["delta"], mdl["gamma"]
    steps = mdl["steps"]

    # 随机流隔离：谣言流 A/B 一致；真相流仅 B 组消耗
    rng_r = np.random.default_rng(cfg["reproducibility"]["rumor_seed"])
    rng_t = np.random.default_rng(cfg["reproducibility"]["truth_seed"])

    n = G.number_of_nodes()
    state = np.full(n, IGNORANT, dtype=np.int8)

    src = pick_seed_node(G, cfg["seed_node"]["strategy"])
    state[src] = R_SPREADER

    t_act = deb["activate_t"] if group == "B" else None
    debunk = pick_debunk_nodes(G, src, deb["count"]) if group == "B" else []

    edges = list(G.edges())  # 固定边序：保证随机数消耗顺序可复现

    rumor_events, truth_events = [], []
    ever_rumor = {src}
    step_counts = []
    converted = 0
    peak_t, peak_s = 0, 1

    for t in range(1, steps + 1):
        if t == t_act:
            # 官方号上线；若此前已在传谣，视为"删帖转辟谣"
            for u in debunk:
                state[u] = T_SPREADER

        # ---- 谣言沿 S-I 边传播（固定边序双向检查；转发倾向取发送者节点行为参数）----
        infections = []
        for u, v in edges:
            su, sv = state[u], state[v]
            if su == R_SPREADER and sv == IGNORANT:
                p = behavior["lambda_node"][u] if behavior is not None else lam
                if rng_r.random() < p:
                    infections.append((u, v))
            elif sv == R_SPREADER and su == IGNORANT:
                p = behavior["lambda_node"][v] if behavior is not None else lam
                if rng_r.random() < p:
                    infections.append((v, u))
        seen = set()
        for parent, child in infections:
            if child in seen:            # 同步内已被感染：按固定边序取第一个感染源
                continue
            seen.add(child)
            state[child] = R_SPREADER
            rumor_events.append((t, int(parent), int(child)))
            ever_rumor.add(int(child))

        # ---- 真相竞争传播：免疫易感者；治愈传谣者 ----
        truth_active = t_act is not None and t >= t_act
        if truth_active:
            immunized, converts = [], []
            for u, v in edges:
                su, sv = state[u], state[v]
                if su == T_SPREADER and sv == IGNORANT and rng_t.random() < delta:
                    immunized.append((u, v))
                elif sv == T_SPREADER and su == IGNORANT and rng_t.random() < delta:
                    immunized.append((v, u))
                if su == T_SPREADER and sv == R_SPREADER and rng_t.random() < gamma:
                    converts.append((u, v))
                elif sv == T_SPREADER and su == R_SPREADER and rng_t.random() < gamma:
                    converts.append((v, u))
            seen_t = set()
            for parent, child in immunized:
                if child in seen_t:
                    continue
                seen_t.add(child)
                state[child] = T_IMMUNE
                truth_events.append((t, int(parent), int(child)))
            seen_c = set()
            for parent, child in converts:
                if child in seen_c or child in seen_t:
                    continue
                seen_c.add(child)
                state[child] = T_SPREADER
                converted += 1
                truth_events.append((t, int(parent), int(child)))  # 转化亦是真相的一次触达

        # ---- 自然沉默：传播者每步以概率 alpha 停止 ----
        stifled = [int(u) for u in np.where(state == R_SPREADER)[0]
                   if rng_r.random() < alpha]
        for u in stifled:
            state[u] = R_STIFLER

        # ---- 逐步统计 ----
        counts = np.bincount(state, minlength=5)
        step_counts.append({"t": t, "I": int(counts[0]), "S": int(counts[1]),
                            "Rn": int(counts[2]), "Rt": int(counts[3]),
                            "T": int(counts[4])})
        if counts[1] > peak_s:
            peak_t, peak_s = t, int(counts[1])

    children = {}
    for _, p, c in rumor_events:
        children.setdefault(p, []).append(c)

    stats = {
        "group": group,
        "n": n,
        "seed_node": int(src),
        "debunk_nodes": [int(u) for u in debunk],
        "activate_t": t_act,
        "params": {"lambda": lam, "alpha": alpha, "delta": delta, "gamma": gamma},
        "behavior": behavior["meta"] if behavior is not None else "uniform(config)",
        "steps": steps,
        "final_counts": step_counts[-1],
        "step_counts": step_counts,
        "ever_heard_rumor": len(ever_rumor),
        "attack_rate": round(len(ever_rumor) / n, 4),
        "rumor_radius": tree_depth(children, int(src)),
        "rumor_transmissions": len(rumor_events),
        "peak_spreaders": peak_s,
        "peak_t": peak_t,
        "converted_to_truth": converted,
        "truth_reached_final": int((state >= T_IMMUNE).sum()),
    }
    return stats, rumor_events, truth_events


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "src" / "config.yaml"
    cfg = load_config(cfg_path)

    G = nx.barabasi_albert_graph(cfg["network"]["n"], cfg["network"]["m"],
                                 seed=cfg["network"]["seed"])
    DATA_DIR.mkdir(exist_ok=True)
    meta = {"n": G.number_of_nodes(), "edges": G.number_of_edges(),
            "avg_degree": round(2 * G.number_of_edges() / G.number_of_nodes(), 2),
            "network_seed": cfg["network"]["seed"]}
    (DATA_DIR / "network_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    results = {}
    behavior = build_node_behavior(G, cfg)
    if behavior is not None:
        print(f"✔ 已注入 LLM 校准节点行为（{behavior['meta']['llm_model']}，"
              f"高素养 {behavior['meta']['forward_rate_literate_raw']} / "
              f"低素养 {behavior['meta']['forward_rate_illiterate_raw']}，缩放 {behavior['meta']['scale']}）")
    else:
        print("ℹ 未找到 data/llm_calibration.json，节点行为取统一参数（config）")
    for group in ("A", "B"):
        stats, rumor_ev, truth_ev = simulate(G, cfg, group, behavior=behavior)
        (DATA_DIR / f"{group}_events.json").write_text(
            json.dumps({"rumor_events": rumor_ev, "truth_events": truth_ev},
                       ensure_ascii=False), encoding="utf-8")
        (DATA_DIR / f"{group}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        results[group] = stats
        label = "无辟谣      " if group == "A" else f"辟谣介入 t={stats['activate_t']}"
        print(f"[{group}] {label}: 感染率 {stats['attack_rate']*100:5.1f}% | "
              f"谣言半径 {stats['rumor_radius']:2d} 环 | "
              f"峰值传播者 {stats['peak_spreaders']}(t={stats['peak_t']}) | "
              f"真相触达 {stats['truth_reached_final']}")

    # ---- 可复现性验证：介入前 A/B 轨迹必须逐事件一致 ----
    pre_t = cfg["debunk"]["activate_t"]
    a_pre = [e for e in json.loads((DATA_DIR / "A_events.json").read_text(encoding="utf-8"))["rumor_events"] if e[0] < pre_t]
    b_pre = [e for e in json.loads((DATA_DIR / "B_events.json").read_text(encoding="utf-8"))["rumor_events"] if e[0] < pre_t]
    print("✔ 可复现性验证通过：A/B 介入前传播轨迹逐事件一致" if a_pre == b_pre
          else "✘ 警告：介入前轨迹不一致，请检查随机流隔离！")


if __name__ == "__main__":
    main()
