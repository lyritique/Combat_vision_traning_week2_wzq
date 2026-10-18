#!/usr/bin/env python3
"""自检：把真值投影回原图，检查它是否落在装甲板上。

为什么需要它
------------
误差表里有一句话："用真值投影去认领检测框"。如果素材与真值不是同一帧，投影会整体偏几十像素，
但最近邻匹配照样能"配上"，最后安静地输出一张看起来完全正常的误差表。
本脚本把这一步显式化：投影必须落在装甲板上，否则报警。

用法
----
    python3 scripts/check_pairs.py --materials Project_materials/含真值图像

可选参数：--pairs config/image_truth_pairs.csv  --out output/pairs  --warn-px 8

依赖：numpy；画核对图另需 Pillow（没有时只输出数值）

输出
----
* 终端表格：每块板的投影中心、最近实测板中心、像素残差、是否可疑
* <out>/<图片名>_check.png：真值四边形 + 实测灯条框
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

# 静态图内参（960x720，零畸变）
FX = FY = 869.1168824543144
CX, CY = 480.0, 360.0
ARMOR_W_MM, ARMOR_H_MM = 135.0, 55.0
COLORS = {"armor_1": (255, 70, 70), "armor_2": (70, 255, 70),
          "armor_3": (90, 170, 255), "armor_4": (255, 80, 255)}


# ------------------------------------------------------------------ md 解析
def parse_md(path):
    """把 图N.md 解析成 {armor: {vis, t, rpy, whn}}，whn = (宽度轴, 高度轴, 法线轴)"""
    poses, axes = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("|") or "armor_" not in s:
                continue
            parts = [p.strip() for p in s.strip("|").split("|")]
            if len(parts) >= 9 and parts[1] in ("正面", "背面"):
                poses[parts[0]] = dict(
                    vis=1 if parts[1] == "正面" else 0,
                    t=(float(parts[2]), float(parts[3]), float(parts[4])),
                    rpy=(float(parts[5]), float(parts[6]), float(parts[7])),
                )
            elif len(parts) == 4 and parts[1].startswith("局部"):
                # | armor_1 | 局部 X | 局部 Z | 局部 Y |  -> 宽 / 高 / 法线
                axes[parts[0]] = (parts[1][-1], parts[2][-1], parts[3][-1])
    return {a: dict(p, whn=axes.get(a, ("X", "Z", "Y"))) for a, p in poses.items()}


def euler_R(rx, ry, rz):
    """R = Rz(yaw) · Ry(pitch) · Rx(roll)，与 md 里的 roll_x/pitch_y/yaw_z 对应。"""
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(P):
    """相机光学系 (m) -> 像素。x 右 / y 下 / z 前。"""
    P = np.atleast_2d(np.asarray(P, float))
    return np.stack([FX * P[:, 0] / P[:, 2] + CX, FY * P[:, 1] / P[:, 2] + CY], axis=1)


def corners(p):
    """按"宽度轴/高度轴"生成 135x55mm 的四个角，再投影到像素。"""
    idx = {"X": 0, "Y": 1, "Z": 2}
    R = euler_R(*p["rpy"])
    wd = R[:, idx[p["whn"][0]]] * (ARMOR_W_MM / 2000.0)
    hd = R[:, idx[p["whn"][1]]] * (ARMOR_H_MM / 2000.0)
    c = np.array(p["t"])
    return project(np.array([c - wd - hd, c + wd - hd, c + wd + hd, c - wd + hd]))


# ---------------------------------------------------------------- 实测灯条
def find_plates(rgb):
    """蓝色灯条连通域 -> 配对 -> 板中心。返回 (bars, plates)，全部是像素坐标。"""
    R, G, B = (rgb[:, :, i].astype(np.int16) for i in range(3))
    m = (B > 110) & (B - R > 35) & (B - G > 5)
    m[: int(0.42 * m.shape[0]), :] = False      # 天空里的蓝色三角形不算

    H, W = m.shape
    seen = np.zeros_like(m, dtype=bool)
    bars = []
    for sy, sx in zip(*np.nonzero(m)):
        if seen[sy, sx]:
            continue
        stack, pts = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pts) < 40:
            continue
        p = np.array(pts)
        h = int(p[:, 0].max() - p[:, 0].min() + 1)
        w = int(p[:, 1].max() - p[:, 1].min() + 1)
        if h < 12 or h < 1.8 * w:                # 灯条是细长的
            continue
        bars.append(dict(cx=float(p[:, 1].mean()), cy=float(p[:, 0].mean()),
                         x0=int(p[:, 1].min()), x1=int(p[:, 1].max()),
                         y0=int(p[:, 0].min()), y1=int(p[:, 0].max()), h=h, w=w))
    bars.sort(key=lambda b: b["cx"])

    used, cand = set(), []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            A, Bb = bars[i], bars[j]
            if min(A["h"], Bb["h"]) / max(A["h"], Bb["h"]) < 0.75:
                continue
            gap = math.hypot(A["cx"] - Bb["cx"], A["cy"] - Bb["cy"])
            ratio = gap / (0.5 * (A["h"] + Bb["h"]))
            if not (1.3 <= ratio <= 3.8):
                continue
            if abs(A["y0"] - Bb["y0"]) > 0.5 * (A["h"] + Bb["h"]):
                continue
            cand.append((abs(ratio - 2.45), i, j))
    cand.sort()

    plates = []
    for _, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        A, Bb = bars[i], bars[j]
        plates.append(((A["cx"] + Bb["cx"]) / 2, (A["cy"] + Bb["cy"]) / 2))
    return bars, plates


def draw(rgb, truth, bars, out_png, title):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    im = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    for b in bars:
        d.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=(255, 210, 0), width=1)
    for armor, p in truth.items():
        col = COLORS.get(armor, (200, 200, 200))
        q = corners(p)
        d.line([tuple(v) for v in q] + [tuple(q[0])], fill=col, width=1)
        u, v = project(p["t"])[0]
        d.line([(u - 8, v), (u + 8, v)], fill=col, width=1)
        d.line([(u, v - 8), (u, v + 8)], fill=col, width=1)
        d.text((u + 10, v - 18), armor[-1] + ("F" if p["vis"] else "b"), fill=col, font=font)
    d.text((6, 4), title, fill=(255, 255, 0), font=font)
    im.save(out_png)
    return True


# -------------------------------------------------------------------- 主流程
def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.lstrip().startswith("#"))]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="Project_materials/含真值图像")
    ap.add_argument("--pairs", default="config/image_truth_pairs.csv")
    ap.add_argument("--out", default="output/pairs")
    ap.add_argument("--bar-px", type=float, default=8.0,
                    help="投影板边到灯条中心多远算『盖住』，默认 8 px")
    args = ap.parse_args()

    if not os.path.isdir(args.materials):
        print("找不到素材目录:", os.path.abspath(args.materials))
        print("用 --materials 指向包含 图1/图2/图3 的『含真值图像』目录")
        return 1

    os.makedirs(args.out, exist_ok=True)
    suspicious = 0
    for row in load_pairs(args.pairs):
        folder, image, md_name = row["folder"].strip(), row["image"].strip(), row["md"].strip()
        img_path = os.path.join(args.materials, folder, image)
        md_path = os.path.join(args.materials, md_name, md_name + ".md")
        if not (os.path.isfile(img_path) and os.path.isfile(md_path)):
            print(f"[跳过] 缺文件: {img_path} 或 {md_path}")
            continue

        truth = parse_md(md_path)
        try:
            from PIL import Image
            rgb = np.asarray(Image.open(img_path).convert("RGB"))
        except ImportError:
            rgb = None

        print(f"\n== {folder}/{image}  vs  {md_name} ==")
        if rgb is None:
            print("   没装 Pillow，只输出投影中心（pip install pillow 可生成核对图）")
            for armor, p in truth.items():
                u, v = project(p["t"])[0]
                print(f"   {armor}: ({u:.1f}, {v:.1f})")
            continue

        bars, plates = find_plates(rgb)
        print(f"   实测：灯条 {len(bars)} 根，配成板 {len(plates)} 块")
        if not bars:
            print("   [注意] 没检出灯条（远距离目标太小），请直接看核对图目视核对")

        suspicious = 0
        for armor, p in truth.items():
            q = corners(p)
            pu, pv = project(p["t"])[0]
            # 两条"竖直边"（分别过 -w 端和 +w 端）的中点，就是两根灯条应该在的地方
            mids = [(q[0] + q[3]) / 2.0, (q[1] + q[2]) / 2.0]
            quad_w = max(math.hypot(*(q[0] - q[1])), math.hypot(*(q[1] - q[2])))

            if bars:
                d = [min(math.hypot(m[0] - b["cx"], m[1] - b["cy"]) for b in bars) for m in mids]
                # 侧视时另一根灯条常被车体挡住，只看较近的一边；正对时两边都要贴住
                side = quad_w < 45.0
                metric = min(d) if side else max(d)
                if metric <= 8.0:
                    verdict = "贴合" + ("（侧视）" if side else "")
                elif metric <= 25.0:
                    verdict = "偏" + ("（侧视）" if side else "")
                else:
                    verdict = "不贴合" + ("（侧视）" if side else "")
                detail = f"两边→灯条 {d[0]:5.1f} / {d[1]:5.1f} px"
            else:
                verdict, detail = "无参照", "未检出灯条"

            flag = ""
            if p["vis"] and verdict.startswith("不贴合"):
                flag = "   <-- 可疑"
                suspicious += 1
            print(f"   {armor}（{'正面' if p['vis'] else '背面'}）"
                  f" 投影=({pu:6.1f},{pv:6.1f})  {detail}  {verdict}{flag}")

        out_png = os.path.join(args.out, f"{os.path.splitext(image)[0]}__vs_{md_name}_check.png")
        if draw(rgb, truth, bars, out_png, f"{image} + {md_name}"):
            print(f"   核对图: {out_png}")

        if suspicious:
            print(f"   [警告] {suspicious} 块正面板的投影没有盖住两根灯条。"
                  f"这不像检测/解算误差，先怀疑素材与真值不是同一帧。")

    print(f"\n完成：{len(load_pairs(args.pairs))} 对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
