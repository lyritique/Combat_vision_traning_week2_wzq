#!/usr/bin/env python3
"""把两种方案的检测结果和真值对比，生成误差表（Markdown + CSV）。

输入:
  - data/truth.csv                   真值（相机系，m，装甲板中心）
  - 若干个 detections CSV（两个程序图片模式输出的文件）：
      image,method,det,label,conf,x_m,y_m,z_m,reproj_px,px,py

用法:
  python3 scripts/evaluate_images.py \
      conventional_approach/output/debug/detections_traditional.csv \
      YOLO_approach/output/debug/detections_yolo.csv

匹配规则:
  用相机内参把真值中心投到像素，和检测框中心（px,py）做一对一最近匹配；
  距离超过 --max-px 的检测记为误检，没被匹配上的可见真值记为漏检。
  误差定义：Δ = 计算值 - 真值（与作业要求一致）。
"""

import argparse
import csv
import math
import os
import sys

# 静态图内参（960x720，零畸变）
FX = FY = 869.1168824543144
CX, CY = 480.0, 360.0


def project(x, y, z):
    return FX * x / z + CX, FY * y / z + CY


def load_truth(path):
    truth = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("image,"):        # 表头
                continue
            image, armor, visible, tx, ty, tz = line.split(",")[:6]
            truth.setdefault(image, []).append(
                dict(armor=armor, visible=int(visible),
                     t=(float(tx), float(ty), float(tz))))
    return truth


def load_detections(paths):
    """读入所有检测 CSV，按 (image, method, det) 去重，后出现的覆盖先出现的。"""
    rows = {}
    for path in paths:
        if not os.path.exists(path):
            print("跳过不存在的文件:", path, file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["image"], r["method"], r["det"])
                rows[key] = r
    return list(rows.values())


def match_image(dets, truthes, max_px):
    """一对一最近匹配，返回 (匹配列表, 未匹配检测, 未匹配真值)。"""
    visible = [t for t in truthes if t["visible"]]
    pairs = []
    for di, d in enumerate(dets):
        dx, dy = float(d["px"]), float(d["py"])
        for ti, t in enumerate(visible):
            u, v = project(*t["t"])
            dist = math.hypot(dx - u, dy - v)
            if dist <= max_px:
                pairs.append((dist, di, ti))
    pairs.sort()

    used_d, used_t, matched = set(), set(), []
    for dist, di, ti in pairs:
        if di in used_d or ti in used_t:
            continue
        used_d.add(di)
        used_t.add(ti)
        matched.append((visible[ti], dets[di], dist))

    missed = [t for ti, t in enumerate(visible) if ti not in used_t]
    false_pos = [d for di, d in enumerate(dets) if di not in used_d]
    return matched, false_pos, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", nargs="+", help="detections_*.csv 路径")
    ap.add_argument("--truth", default="data/truth.csv")
    ap.add_argument("--out-prefix", default="output/eval_images")
    ap.add_argument("--max-px", type=float, default=80.0,
                    help="检测中心与真值投影的最大匹配距离（像素）")
    args = ap.parse_args()

    truth = load_truth(args.truth)
    dets = load_detections(args.detections)

    by_image = {}
    for d in dets:
        by_image.setdefault(d["image"], []).append(d)

    table, summary = [], {}
    for image in sorted(truth):
        if image not in by_image:
            print("没有检测结果:", image, file=sys.stderr)
            continue
        print(f"\n== {image} ==")

        # 两种方案各自独立做一对一匹配（否则同一块真值会被一个方案"抢走"）
        by_method = {}
        for d in by_image[image]:
            by_method.setdefault(d["method"], []).append(d)

        for method, mdets in sorted(by_method.items()):
            matched, false_pos, missed = match_image(mdets, truth[image], args.max_px)

            for t, d, dist in sorted(matched, key=lambda m: m[0]["armor"]):
                cx, cy, cz = float(d["x_m"]), float(d["y_m"]), float(d["z_m"])
                tx, ty, tz = t["t"]
                dx, dy, dz = cx - tx, cy - ty, cz - tz
                err3 = math.sqrt(dx * dx + dy * dy + dz * dz)
                table.append(dict(image=image, method=method, armor=t["armor"],
                                  x=cx, y=cy, z=cz, tx=tx, ty=ty, tz=tz,
                                  dx=dx, dy=dy, dz=dz, err3=err3,
                                  reproj=float(d["reproj_px"]),
                                  label=d["label"], conf=float(d["conf"]),
                                  px_match=dist))
                summary.setdefault(method, []).append(err3)
                print(f"  {method:11s} {t['armor']}: Δ=({dx:+.3f},{dy:+.3f},{dz:+.3f}) "
                      f"三维={err3:.3f}m 重投影={float(d['reproj_px']):.2f}px 像素差={dist:.1f}px")

            for d in false_pos:
                print(f"  {method} 误检（未匹配）: label={d['label']} conf={d['conf']}")
            for t in missed:
                print(f"  {method} 漏检（可见但没检出）: {t['armor']}")

    # ---- 输出 CSV ----
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "method", "armor", "x", "y", "z", "tx", "ty", "tz",
                    "dx", "dy", "dz", "err3d", "reproj_px", "label", "conf"])
        for r in table:
            w.writerow([r["image"], r["method"], r["armor"],
                        f"{r['x']:.6f}", f"{r['y']:.6f}", f"{r['z']:.6f}",
                        f"{r['tx']:.6f}", f"{r['ty']:.6f}", f"{r['tz']:.6f}",
                        f"{r['dx']:+.4f}", f"{r['dy']:+.4f}", f"{r['dz']:+.4f}",
                        f"{r['err3']:.4f}", f"{r['reproj']:.2f}", r["label"], f"{r['conf']:.4f}"])

    # ---- 输出 Markdown ----
    md_path = args.out_prefix + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 图 | 方案 | 板号 | 计算 (x,y,z) m | 真值 (x,y,z) m | Δx | Δy | Δz | 三维误差 | 重投影 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in table:
            f.write(f"| {r['image']} | {r['method']} | {r['armor']} | "
                    f"({r['x']:.3f}, {r['y']:.3f}, {r['z']:.3f}) | "
                    f"({r['tx']:.3f}, {r['ty']:.3f}, {r['tz']:.3f}) | "
                    f"{r['dx']:+.3f} | {r['dy']:+.3f} | {r['dz']:+.3f} | "
                    f"**{r['err3']:.3f}** | {r['reproj']:.2f} px |\n")

    print("\n== 汇总（平均三维误差）==")
    for method, errs in sorted(summary.items()):
        print(f"  {method:11s}: {sum(errs)/len(errs):.3f} m（{len(errs)} 块）")
    print("\n已写出:", csv_path, "和", md_path)


if __name__ == "__main__":
    main()
