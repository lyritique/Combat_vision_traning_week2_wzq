#!/usr/bin/env python3
"""把两种方案的视频 CSV 汇总成报告用的表格，重点是**重投影误差的分布**。

用法:
    python3 scripts/video_stats.py \
        conventional_approach/output/traditional_video.csv \
        YOLO_approach/output/yolo_video.csv --frames 531

可选: --frames 总帧数（默认 531，用来算帧检出率；CSV 里没有检测的帧不写行，所以没法从 CSV 推出来）

输出的 Markdown 表格可以直接粘进报告。两列 CSV 的列名不同，脚本按列名取，不依赖顺序：
    传统: frame,bars,candidates,kept,number,conf,x_m,y_m,z_m,reproj_px
    YOLO: frame,dets,class,color,conf,pose_ok,x_m,y_m,z_m,reproj_px
"""
import argparse
import collections
import csv
import os
import statistics
import sys


def pctl(vals, q):
    """最近秩分位数（不插值，报告里更好解释）。"""
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[k]


def load(path):
    # utf-8-sig：兼容 Windows 上用记事本/Excel 另存出来的带 BOM 的 CSV
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def label_col(rows):
    for key in ("number", "class"):
        if rows and key in rows[0]:
            return key
    return None


def summarize(name, rows, total_frames):
    if not rows:
        return None
    frames = [int(r["frame"]) for r in rows]
    per_frame = collections.Counter(frames)
    rp = [float(r["reproj_px"]) for r in rows]
    zs = [float(r["z_m"]) for r in rows]

    # 每帧"最坏的那一块"的重投影，再取中位：反映最差情况的稳定性
    worst = {}
    for r, f in zip(rows, frames):
        worst[f] = max(worst.get(f, 0.0), float(r["reproj_px"]))
    worst_vals = list(worst.values())

    lc = label_col(rows)
    labels = collections.Counter(r[lc] for r in rows) if lc else collections.Counter()

    return dict(
        name=name, n=len(rows), frames=len(per_frame),
        rate=100.0 * len(per_frame) / total_frames if total_frames else float("nan"),
        per_frame_mean=len(rows) / len(per_frame),
        rp_mean=statistics.fmean(rp), rp_med=statistics.median(rp),
        rp_p95=pctl(rp, 0.95), rp_max=max(rp),
        worst_med=statistics.median(worst_vals),
        z_min=min(zs), z_med=statistics.median(zs), z_max=max(zs),
        multi=sum(1 for v in per_frame.values() if v >= 2),
        labels=labels.most_common(3),
    )


def histogram(vals, edges=(0.1, 0.3, 0.5, 1.0, 2.0, 5.0)):
    """重投影误差分布（占比 %）。"""
    n = len(vals)
    out, prev = [], 0.0
    for e in edges:
        c = sum(1 for v in vals if prev <= v < e)
        out.append((f"[{prev:g},{e:g})", 100.0 * c / n if n else 0.0))
        prev = e
    c = sum(1 for v in vals if v >= prev)
    out.append((f">={prev:g}", 100.0 * c / n if n else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="视频结果 CSV")
    ap.add_argument("--frames", type=int, default=531, help="视频总帧数")
    ap.add_argument("--labels", default="传统视觉,神经网络", help="方案名，按顺序对应 CSV")
    args = ap.parse_args()

    names = [s.strip() for s in args.labels.split(",")]
    results = []
    for k, path in enumerate(args.csvs):
        if not os.path.isfile(path):
            print("找不到:", path, file=sys.stderr)
            continue
        rows = load(path)
        name = names[k] if k < len(names) else os.path.basename(path)
        s = summarize(name, rows, args.frames)
        if s:
            s["rows"] = rows
            results.append(s)

    if not results:
        print("没有可分析的数据", file=sys.stderr)
        return 1

    # ---------- 表 1：总览 ----------
    print("### 视频结果总览\n")
    print("| 方案 | 检测块数 | 涉及帧 | 帧检出率 | 每帧均值 | 重投影 均值 | 中位 | P95 | 最大 | 帧内最坏(中位) | z 中位 | z 范围 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in results:
        print(f"| {s['name']} | {s['n']} | {s['frames']}/{args.frames} | {s['rate']:.2f}% | "
              f"{s['per_frame_mean']:.2f} | {s['rp_mean']:.3f} px | {s['rp_med']:.3f} px | "
              f"{s['rp_p95']:.3f} px | {s['rp_max']:.2f} px | {s['worst_med']:.3f} px | "
              f"{s['z_med']:.2f} m | {s['z_min']:.2f} ~ {s['z_max']:.2f} m |")

    # ---------- 表 2：重投影误差分布 ----------
    print("\n### 重投影误差分布（占比）\n")
    edges = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
    head = histogram([0.0], edges)  # 只为拿分桶名
    print("| 方案 | " + " | ".join(h[0] for h in head) + " |")
    print("|---" * (len(head) + 1) + "|")
    for s in results:
        vals = [float(r["reproj_px"]) for r in s["rows"]]
        print(f"| {s['name']} | " + " | ".join(f"{p:.1f}%" for _, p in histogram(vals, edges)) + " |")

    # ---------- 表 3：每帧检测块数 ----------
    print("\n### 每帧检测块数分布\n")
    print("| 方案 | 1 块 | 2 块 | ≥3 块 | 说明 |")
    print("|---|---|---|---|---|")
    for s in results:
        c = collections.Counter()
        for r in s["rows"]:
            c[int(r["frame"])] += 1
        one = sum(1 for v in c.values() if v == 1)
        two = sum(1 for v in c.values() if v == 2)
        more = sum(1 for v in c.values() if v >= 3)
        print(f"| {s['name']} | {one} | {two} | {more} | 双块帧 = 一帧里同时框到两块板 |")

    # ---------- 类别/颜色 ----------
    print("\n### 类别 / 号码分布\n")
    for s in results:
        print(f"- **{s['name']}**: " + "，".join(f"{k}={v}" for k, v in s["labels"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
