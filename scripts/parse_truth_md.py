#!/usr/bin/env python3
"""从 图N.md 解析真值，生成 data/truth.csv。

与旧版的区别
------------
* 对应关系不再写死在代码里的 MD_TO_IMAGE 字典，而是读 config/image_truth_pairs.csv：
  一个图片 -> 一份真值，`enabled` 控制是否进误差表。改对应关系不用改代码。
* 素材目录可以传参（--materials），不再假设仓库根目录下有 Project_materials/。
* 图片文件不存在时报错并跳过，不再静默生成一张缺行的表。

用法:
    python3 scripts/parse_truth_md.py --materials Project_materials/含真值图像

输出: data/truth.csv（列：image,armor,visible,tx,ty,tz；相机光学系，单位 m，参考点=装甲板中心）
"""
import argparse
import csv
import os
import re
import sys

ROW_RE = re.compile(r"^\|\s*(armor_\d+)\s*\|\s*(正面|背面)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|")


def parse_md(path):
    """只取 1.1 节（相机光学系）的 pose 表：armor / 可见 / tx / ty / tz。"""
    rows, in_table = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("| 板 ") or line.startswith("|板|"):
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if not in_table:
                continue
            m = ROW_RE.match(line)
            if m:
                armor, face, tx, ty, tz = m.groups()
                rows.append((armor.strip(), 1 if face == "正面" else 0,
                             float(tx), float(ty), float(tz)))
            elif line.strip() and not line.startswith("|"):
                in_table = False
    return rows


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(l for l in f if not l.lstrip().startswith("#"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="Project_materials/含真值图像")
    ap.add_argument("--pairs", default="config/image_truth_pairs.csv")
    ap.add_argument("--out", default="data/truth.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.materials):
        print("找不到素材目录:", os.path.abspath(args.materials), file=sys.stderr)
        print("用 --materials 指向包含 图1/图2/图3 的『含真值图像』目录", file=sys.stderr)
        return 1
    if not os.path.isfile(args.pairs):
        print("找不到配置:", args.pairs, file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total, skipped = 0, []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write("# 装甲板真值（相机光学系，单位 m；参考点 = 装甲板几何中心）\n")
        f.write("# 由 scripts/parse_truth_md.py 按 config/image_truth_pairs.csv 生成，勿手改\n")
        f.write("# 列：image,armor,visible,tx,ty,tz\n")
        w = csv.writer(f)

        for row in load_pairs(args.pairs):
            folder, image, md = row["folder"].strip(), row["image"].strip(), row["md"].strip()
            if row["enabled"].strip() not in ("1", "true", "yes"):
                skipped.append(f"{folder}/{image}（enabled=0：{row['note'].strip()}）")
                continue
            img_path = os.path.join(args.materials, folder, image)
            md_path = os.path.join(args.materials, md, md + ".md")
            if not os.path.isfile(img_path):
                print(f"[跳过] 图片不存在: {img_path}", file=sys.stderr)
                skipped.append(f"{folder}/{image}（文件不存在）")
                continue
            if not os.path.isfile(md_path):
                print(f"[跳过] 真值不存在: {md_path}", file=sys.stderr)
                continue

            rows = parse_md(md_path)
            if not rows:
                print(f"[跳过] 没解析到内容: {md_path}", file=sys.stderr)
                continue
            f.write(f"# {md} -> {image}\n")
            for armor, visible, tx, ty, tz in rows:
                w.writerow([image, armor, visible, f"{tx:+.6f}", f"{ty:+.6f}", f"{tz:+.6f}"])
                total += 1
            print(f"{md} -> {image}：{len(rows)} 块")

    print(f"\n共写入 {total} 行 -> {args.out}")
    if skipped:
        print("未计入（如需纳入，改 config/image_truth_pairs.csv 的 enabled）：")
        for s in skipped:
            print("  - " + s)
    print("提示：换素材后先跑 scripts/check_pairs.py，确认真值投影落在装甲板上再生成误差表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
