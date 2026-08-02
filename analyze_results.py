#!/usr/bin/env python3
"""
ReasonLite 实验分析: 验证集多维度统计 + 训练 loss 提取
用法:
  python analyze_results.py                       # 验证集分析 -> result/autodl/analysis.md
  python analyze_results.py --train-log result/pipeline_output.log   # Mac MLX 训练日志
  python analyze_results.py --train-log logs/train.log               # AutoDL Trainer 日志
输出:
  result/autodl/analysis.md    验证集: 正确率/题型分层/作答质量/稳定性/部分分
  result/autodl/train_loss.md  每实验 loss 变化
"""
import os, re, json, sys, argparse
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE, "result", "autodl")

MATRIX = [
    ("baseline", "—", None, "baseline", False),
    ("LoRA", "r16 后8层", 100, "lora_100", True),
    ("LoRA", "r16 后8层", 500, "lora_500", True),
    ("LoRA", "r16 后8层", 1000, "lora_1000", True),
    ("Full", "后8层", 100, "full8_100", False),
    ("Full", "后8层", 500, "full8_500", False),
    ("Full", "后8层", 1000, "full8_1000", False),
    ("Full", "后16层", 100, "full16_100", False),
    ("Full", "后16层", 500, "full16_500", False),
    ("Full", "后16层", 1000, "full16_1000", False),
]
NAMES = [n for _, _, _, n, _ in MATRIX]


# ---------- 题型分类 (数学竞赛视角) ----------
def classify(expected):
    e = expected.strip()
    if re.search(r'\\frac|\\dfrac|\\sqrt', e):
        return "分数/根式"
    if re.search(r'[a-zA-Z]', e):
        return "代数/符号"
    if re.search(r'\\text|\\begin\{cases\}|%|\(|\)|\{', e):
        return "特殊格式"  # 集合/区间/百分比/方程组
    if re.search(r'^-?\d+\.?\d*$', e):
        return "纯数字"
    return "其他"


def num_value(expr):
    """尝试把 latex/数字转数值"""
    try:
        return float(expr)
    except ValueError:
        pass
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        v = simplify(parse_latex(expr)).evalf()
        return float(v)
    except Exception:
        return None


# ---------- 验证集统计 ----------
def analyze_eval():
    data = {}
    for n in NAMES:
        p = os.path.join(EVAL_DIR, f"{n}.json")
        if os.path.exists(p):
            data[n] = json.load(open(p))
    if not data:
        print("未找到 result/autodl/*.json")
        return

    total = data[NAMES[0]]["total"]
    lines = ["# ReasonLite 验证集分析 (50 题数学竞赛)", ""]

    # 1. 总表
    lines += ["## 1. 各版本总览", "",
              "| 版本 | 正确率 | 平均token | 每分token | \\boxed率 | 输出含思考 | 提取失败 |",
              "|------|-------|----------|----------|----------|-----------|---------|"]
    rows = []
    for n in NAMES:
        if n not in data:
            continue
        d = data[n]
        toks = [r["completion_tokens"] for r in d["results"]]
        think = sum(1 for r in d["results"] if "<think" in r["output"])
        acc = d["accuracy"]
        rows.append((acc, n, d, toks, think))
    for acc, n, d, toks, think in sorted(rows, key=lambda x: -x[0]):
        tpc = round(sum(toks) / len(toks) / acc, 1) if acc > 0 else float("inf")
        lines.append(f"| {n} | {acc*100:.1f}% | {d['avg_completion_tokens']} | {tpc} | {d['boxed_ratio']*100:.0f}% | {think}/{len(toks)} | {d['extract_failed']} |")

    # 2. 题型分层
    lines += ["", "## 2. 题型分层正确率 (数值/分数根式/代数符号/特殊格式)", "",
              "| 版本 | " + " | ".join(["纯数字", "分数/根式", "代数/符号", "特殊格式"]) + " | 全类平均 |",
              "|------|" + "------|" * 5]
    for n in NAMES:
        if n not in data:
            continue
        d = data[n]
        grp = defaultdict(list)
        for r in d["results"]:
            grp[classify(r["expected"])].append(r["correct"])
        cells = []
        for t in ["纯数字", "分数/根式", "代数/符号", "特殊格式"]:
            g = grp.get(t, [])
            cells.append(f"{sum(g)}/{len(g)}" if g else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {d['accuracy']*100:.1f}% |")

    # 3. 稳定性: 每题跨版本正确数
    lines += ["", "## 3. 稳定性 (每题在 10 版本中被答对数)", "",
              "| 答对版本数 | 题数 | 题目 |",
              "|-----------|------|------|"]
    per_q = defaultdict(int)
    q_meta = {}
    for i in range(total):
        cnt = 0
        for n in data:
            if data[n]["results"][i]["correct"]:
                cnt += 1
        per_q[cnt] += 1
        q_meta[i] = cnt
    for k in sorted(per_q, reverse=True):
        idxs = [i + 1 for i in range(total) if q_meta[i] == k]
        tag = " (全对)" if k == len(data) else " (全错)" if k == 0 else ""
        shown = ",".join(str(x) for x in idxs[:12]) + ("..." if len(idxs) > 12 else "")
        lines.append(f"| {k} | {per_q[k]} | 题{shown}{tag} |")

    # 4. 部分分: 数值接近度 (wrong 且数值型)
    lines += ["", "## 4. 部分分统计 (数值型题目答错时的接近程度)", "",
              "| 版本 | 数值型题数 | 答对 | 答错 | 错但|Δ|<0.5 | 错但|Δ|<2 | 完全跑偏 |",
              "|------|-----------|------|------|------------|------------|----------|"]
    for acc, n, d, toks, think in sorted(rows, key=lambda x: -x[0]):
        n_num = n_close1 = n_close2 = n_wrong = n_correct = 0
        for r in d["results"]:
            if classify(r["expected"]) != "纯数字":
                continue
            n_num += 1
            ev = num_value(r["expected"])
            if r["correct"]:
                n_correct += 1
                continue
            pv = num_value(r["extracted"])
            if ev is not None and pv is not None:
                n_wrong += 1
                diff = abs(pv - ev)
                if diff < 0.5:
                    n_close1 += 1
                if diff < 2:
                    n_close2 += 1
        lines.append(f"| {n} | {n_num} | {n_correct} | {n_wrong} | {n_close1} | {n_close2} | {n_wrong - n_close2} |")

    lines += ["", "注: 部分分统计仅含 expected 为纯数字的题目; |Δ| 为模型答案与标准答案数值差(答错样本)。"]
    path = os.path.join(EVAL_DIR, "analysis.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[分析报告 -> {path}]")


# ---------- 训练 loss 提取 ----------
def parse_train_log(path):
    """解析训练日志, 返回 {experiment: [(iter, train_loss|None, val_loss|None)]}"""
    exps = {}
    cur = None
    exp_re = re.compile(r"->\s*([\w]+)\s*$")
    mlx_train = re.compile(r"Iter (\d+): Train loss ([\d.eE+-]+)")
    mlx_val = re.compile(r"Iter (\d+): Val loss ([\d.eE+-]+)")
    hf_loss = re.compile(r"'loss':\s*([\d.eE+-]+)")          # Trainer dict
    hf_eval = re.compile(r"'eval_loss':\s*([\d.eE+-]+)")     # Trainer eval

    with open(path) as f:
        idx = 0  # HF Trainer 日志无 step 字段, 用顺序号近似
        for line in f:
            m = exp_re.search(line)
            if m:
                cur = m.group(1)
                exps.setdefault(cur, [])
                idx = 0
                continue
            if cur is None:
                continue
            tl = mlx_train.search(line) or hf_loss.search(line)
            vl = mlx_val.search(line) or hf_eval.search(line)
            if tl or vl:
                it = mlx_train.search(line) or mlx_val.search(line)
                step = int(it.group(1)) if it else idx
                idx += 1
                exps[cur].append((step, float(tl.group(1)) if tl else None, float(vl.group(1)) if vl else None))
    return exps


def report_loss(path):
    exps = parse_train_log(path)
    if not exps:
        print(f"未从 {path} 解析到 loss 数据")
        return
    lines = [f"# 训练 loss 变化 (来源: {os.path.basename(path)})", "",
             "采样展示: train loss / val loss 随 step 变化", ""]
    for name, pts in exps.items():
        lines += [f"## {name}", "", "| step | train loss | val loss |", "|------|-----------|---------|"]
        # 采样: 全部列出(小实验)或每 10 条一条
        step_show = pts if len(pts) <= 50 else pts[::max(1, len(pts) // 50)]
        for s, tl, vl in step_show:
            lines.append(f"| {s} | {tl if tl is not None else '-'} | {vl if vl is not None else '-'} |")
        lines.append("")
    path_out = os.path.join(EVAL_DIR, "train_loss.md")
    open(path_out, "w").write("\n".join(lines))
    print(f"[训练 loss -> {path_out}]")
    for name, pts in exps.items():
        tl = [p for p in pts if p[1] is not None]
        vl = [p for p in pts if p[2] is not None]
        print(f"  {name}: train {tl[0][1]:.3f}->{tl[-1][1]:.3f} ({len(tl)}点) | val {vl[0][2]:.3f}->{vl[-1][2]:.3f}" if tl and vl
              else f"  {name}: 数据不足")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default=None, help="训练日志路径 (Mac MLX 或 AutoDL Trainer 格式)")
    args = ap.parse_args()
    analyze_eval()
    if args.train_log:
        report_loss(args.train_log)
