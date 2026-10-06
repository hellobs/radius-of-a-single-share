# -*- coding: utf-8 -*-
"""
Layer 2 —— LLM 多智能体行为校准实验（MAVIS 思路 / 本地大模型）
================================================================
24 名带人设的虚拟网民（媒介素养 × 情绪 × 社交习惯 × 身份）遭遇同一条
**虚构**校园谣言（仅作仿真刺激材料），经本地 Ollama 大模型做两轮决策：

  第一轮  仅见谣言     → 测得不同媒介素养人群的转发率差异
                         （→ 网络仿真中节点传播倾向 λ 的相对差异）
  第二轮  见官方通报   → 测得"坚持传谣 / 删帖 / 转发辟谣 / 不再理会"分布
                         （→ 治愈转化率 γ、真相转发率 δ 的相对差异）

输出 data/llm_calibration.json，供 simulate.py 将行为参数按素养分型
注入 3000 节点网络（总量守恒缩放，保留网络动力学节奏）。

用法
  python src/llm_calibration.py          # 真实调用本地大模型
  python src/llm_calibration.py --dry    # 只打印人设与提示词，不调用
"""
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import DATA_DIR, load_config  # noqa: E402

# ---- 仿真刺激材料（完全虚构，不指代任何真实学校/事件）----
RUMOR = ("【网传】临江大学二食堂后厨被曝使用合成肉，已有数十名学生食物中毒住院，"
         "校方疑似封锁消息！转起来让更多人看到！")
REBUTTAL = ("【临江大学官方通报】经属地市场监管部门与校方联合核查：二食堂本周食材抽检"
            "全部合格，网传\u201c合成肉\u201d\u201c学生集体中毒\u201d均为不实信息，"
            "首发造谣账号已被依法处置。请师生不信谣、不传谣。")

TRAITS = [("高素养", "接受过系统的媒介素养训练，习惯核查信源、等待官方通报"),
          ("低素养", "没有接受过媒介素养训练，主要凭直觉和情绪判断信息")]
EMOTIONS = [("冲动", "情绪容易被煽动，看到愤怒的内容会跟着激动"),
            ("冷静", "遇事冷静，不容易被情绪带节奏")]
SOCIALS = [("爱转发", "热衷转发热点，是朋友圈的活跃分子"),
           ("少转发", "很少转发内容，习惯潜水围观")]
IDENTITIES = ["大一新生", "理工科研究生", "互联网公司上班族", "中学教师",
              "社区超市店员", "自由职业摄影师", "医学院学生", "快递站点小哥",
              "银行职员", "美食自媒体博主", "退休返聘会计", "健身教练"]

SYSTEM_PROMPT = ("你是多智能体社会仿真中的一个虚拟社交网络用户智能体。"
                 "请完全代入给定人设进行决策。只输出一个 JSON 对象，"
                 "禁止输出任何其他文字或解释。")

ROUND1_ACTIONS = "forward=转发扩散 / question=质疑求证 / rebut=公开辟谣 / ignore=不回应"
ROUND2_ACTIONS = "keep_forward=坚持传谣不删 / delete=删除转发 / forward_rebuttal=转发官方辟谣 / ignore=不再理会"

VALID_R1 = {"forward", "question", "rebut", "ignore"}
VALID_R2 = {"keep_forward", "delete", "forward_rebuttal", "ignore"}


def build_personas(n):
    """确定性构造 n 个均衡分型的虚拟网民人设。"""
    personas = []
    for i in range(n):
        t = TRAITS[i % 2]
        e = EMOTIONS[(i // 2) % 2]
        s = SOCIALS[(i // 4) % 2]
        ident = IDENTITIES[i % len(IDENTITIES)]
        personas.append({
            "id": i,
            "identity": ident,
            "literacy": t[0],
            "emotion": e[0],
            "social": s[0],
            "desc": f"你是一名{ident}。{t[1]}。你{e[1]}。平时{s[1]}。",
        })
    return personas


def http_json(url, payload=None, timeout=90):
    """GET(payload=None) / POST(payload=dict)，返回解析后的 JSON。"""
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(cfg):
    try:
        data = http_json(cfg["llm"]["base_url"].rstrip("/") + "/models", timeout=8)
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return None


def chat_once(cfg, messages):
    body = {"model": cfg["llm"]["model"], "messages": messages,
            "temperature": cfg["llm"]["temperature"], "max_tokens": cfg["llm"]["max_tokens"]}
    data = http_json(cfg["llm"]["base_url"].rstrip("/") + "/chat/completions",
                     payload=body, timeout=120)
    return data["choices"][0]["message"]["content"]


def parse_action(text, valid):
    """从模型输出中稳健地提取 {action, confidence, reason}。"""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            act = str(obj.get("action", "")).strip().lower()
            if act in valid:
                conf = obj.get("confidence", 0.5)
                try:
                    conf = max(0.0, min(1.0, float(conf)))
                except (TypeError, ValueError):
                    conf = 0.5
                return {"action": act, "confidence": conf,
                        "reason": str(obj.get("reason", ""))[:60]}
        except json.JSONDecodeError:
            pass
    low = text.lower()
    for act in valid:  # 关键词兜底
        if act in low:
            return {"action": act, "confidence": 0.5, "reason": "(keyword fallback)"}
    return None


def ask_agent(cfg, persona, user_prompt, valid, tries=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    for _ in range(tries):
        try:
            out = parse_action(chat_once(cfg, messages), valid)
            if out is not None:
                out["parse_ok"] = True
                return out
        except Exception as exc:  # 网络/服务波动重试
            last_err = str(exc)
    return {"action": "ignore", "confidence": 0.0, "reason": f"(failed: {locals().get('last_err','?')[:60]})",
            "parse_ok": False}


def rate(rows, action):
    return round(sum(1 for r in rows if r["action"] == action) / max(1, len(rows)), 4)


def aggregate(round1, round2_by_id, personas):
    lit = {p["id"]: p["literacy"] for p in personas}
    fwd = [r for r in round1 if r["action"] == "forward"]
    fwd_hi = [r for r in fwd if lit[r["id"]] == "高素养"]
    fwd_lo = [r for r in fwd if lit[r["id"]] == "低素养"]

    r2 = list(round2_by_id.values())
    keep = [r for r in r2 if r["action"] == "keep_forward"]
    keep_hi = [r for r in r2 if r["action"] == "keep_forward" and lit[r["id"]] == "高素养"]
    keep_lo = [r for r in r2 if r["action"] == "keep_forward" and lit[r["id"]] == "低素养"]

    agg = {
        "round1": {
            "forward_rate": rate(round1, "forward"),
            "forward_rate_literate": rate([r for r in round1 if lit[r["id"]] == "高素养"], "forward"),
            "forward_rate_illiterate": rate([r for r in round1 if lit[r["id"]] == "低素养"], "forward"),
            "question_rate": rate(round1, "question"),
            "rebut_rate": rate(round1, "rebut"),
            "ignore_rate": rate(round1, "ignore"),
        },
        "round2": {
            "respondents": len(r2),
            "keep_rate": rate(r2, "keep_forward"),
            "delete_rate": rate(r2, "delete"),
            "rebut_forward_rate": rate(r2, "forward_rebuttal"),
            "ignore_rate": rate(r2, "ignore"),
        },
        "estimates": {
            "lambda_ratio_lo_hi": None,
            "gamma_overall": round(1 - rate(r2, "keep_forward"), 4),
            "gamma_literate": round(1 - rate([x for x in r2 if lit[x["id"]] == "高素养"], "keep_forward"), 4) if fwd_hi else None,
            "gamma_illiterate": round(1 - rate([x for x in r2 if lit[x["id"]] == "低素养"], "keep_forward"), 4) if fwd_lo else None,
            "delta_proxy": rate(r2, "forward_rebuttal"),
        },
    }
    hi_r, lo_r = agg["round1"]["forward_rate_literate"], agg["round1"]["forward_rate_illiterate"]
    agg["estimates"]["lambda_ratio_lo_hi"] = round(lo_r / hi_r, 2) if hi_r > 0 else None
    return agg


def main():
    dry = "--dry" in sys.argv
    cfg = load_config()
    personas = build_personas(cfg["llm"]["personas"])

    if dry:
        for p in personas:
            print(f"[{p['id']:02d}] {p['desc']}")
        print("\n--- 第一轮提示词示例 ---")
        print(f"{personas[0]['desc']}\n\n你在社交平台刷到热帖：\n{RUMOR}\n\n请决策：{ROUND1_ACTIONS}")
        return

    models = list_models(cfg)
    if not models:
        print("✘ 无法连接本地大模型服务。请先启动：ollama serve（或 Ollama 桌面端）")
        sys.exit(1)
    print(f"✔ 服务在线，可用模型：{models}")
    if cfg["llm"]["model"] not in models:
        print(f"✘ 配置的模型 {cfg['llm']['model']} 不在列表中，请修改 config.yaml")
        sys.exit(1)

    round1 = []
    for p in personas:
        prompt = (f"{p['desc']}\n\n你在社交平台刷到热帖：\n{RUMOR}\n\n"
                  f"请决策（action 取值）：{ROUND1_ACTIONS}\n"
                  '输出：{"action": "...", "confidence": 0~1, "reason": "一句话"}')
        out = ask_agent(cfg, p, prompt, VALID_R1)
        out.update({"id": p["id"], "literacy": p["literacy"]})
        round1.append(out)
        print(f"  R1 [{p['id']:02d}] {p['identity']}/{p['literacy']} -> {out['action']} ({out['confidence']:.2f})")

    forwarders = [p for p, r in zip(personas, round1) if r["action"] == "forward"]
    round2_by_id = {}
    for p in forwarders:
        prompt = (f"{p['desc']}\n\n一小时前你转发了：\n{RUMOR}\n\n现在官方发布通报：\n{REBUTTAL}\n\n"
                  f"请决策（action 取值）：{ROUND2_ACTIONS}\n"
                  '输出：{"action": "...", "confidence": 0~1, "reason": "一句话"}')
        out = ask_agent(cfg, p, prompt, VALID_R2)
        out.update({"id": p["id"], "literacy": p["literacy"]})
        round2_by_id[p["id"]] = out
        print(f"  R2 [{p['id']:02d}] {p['identity']}/{p['literacy']} -> {out['action']} ({out['confidence']:.2f})")

    agg = aggregate(round1, round2_by_id, personas)
    payload = {
        "model": cfg["llm"]["model"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_personas": len(personas),
        "rumor_text": RUMOR,
        "rebuttal_text": REBUTTAL,
        "round1": round1,
        "round2": list(round2_by_id.values()),
        "aggregates": agg,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "llm_calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    a1 = agg["round1"]
    a2 = agg["round2"]
    est = agg["estimates"]
    print("\n==== LLM 校准实验结果 ====")
    print(f"一轮转发率: 全体 {a1['forward_rate']:.2f} | 高素养 {a1['forward_rate_literate']:.2f}"
          f" / 低素养 {a1['forward_rate_illiterate']:.2f}（比值 {est['lambda_ratio_lo_hi']}）")
    print(f"质疑 {a1['question_rate']:.2f} | 辟谣 {a1['rebut_rate']:.2f} | 忽略 {a1['ignore_rate']:.2f}")
    print(f"二轮(n={a2['respondents']}): 坚持传谣 {a2['keep_rate']:.2f} | 删帖 {a2['delete_rate']:.2f}"
          f" | 转发辟谣 {a2['rebut_forward_rate']:.2f} | 不再理会 {a2['ignore_rate']:.2f}")
    print(f"估计: γ(转化)={est['gamma_overall']:.2f}（高素养 {est['gamma_literate']} / 低素养 {est['gamma_illiterate']}）"
          f" | δ代理(转发辟谣)={est['delta_proxy']:.2f}")
    print(f"已保存 -> {DATA_DIR / 'llm_calibration.json'}")
    print("下一步：python src/simulate.py 将自动读取本文件，按素养分型注入节点行为参数。")


if __name__ == "__main__":
    main()
