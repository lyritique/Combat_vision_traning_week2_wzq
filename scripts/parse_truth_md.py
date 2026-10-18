#!/usr/bin/env python3
"""从作业材料里的 图N.md 解析真值，生成 data/truth.csv。

数据来源：Project_materials/含真值图像/图N/图N.md 的 1.1 节
  「装甲板 6D 真值位姿（相机光学系，单位 m / 度）」表格
  取 tx, ty, tz 和「可见」列（正面=1，背面=0）。

用法:
    python3 scripts/parse_truth_md.py

图片和 md 的对应关系写在 MD_TO_IMAGE 里。图2 目前待确认（目录里的三张
jpg 检测中心都和图2.md 的投影对不上），确认后把它加进映射即可。
"""

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALS = os.path.join(ROOT, "Project_materials", "含真值图像")
OUT = os.path.join(ROOT, "data", "truth.csv")

# md 文件夹 -> 对应的原图文件名（做误差表时按这个 key 匹配检测结果）
MD_TO_IMAGE = {
    "图1": "Image_1791695492312_183.jpg",
    "图3": "Image_1791695502223_335.jpg",
    "图2": "Image_1791732303841_547.jpg",   # 待确认：目录里三张图的检测中心都和图2.md 投影对不上
}

# | armor_1 | 正面 | -0.174340 | +0.169600 | +3.794632 | ... |
ROW_RE = re.compile(r"^\|\s*(armor_\d+)\s*\|\s*(正面|背面)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|")


def parse_md(path):
    rows = []
    in_pose_table = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("| 板 ") or line.startswith("|板|"):
                in_pose_table = True
                continue
            if in_pose_table and line.startswith("|---"):
                continue
            if not in_pose_table:
                continue
            m = ROW_RE.match(line)
            if m:
                armor, face, tx, ty, tz = m.groups()
                rows.append((armor.strip(), 1 if face == "正面" else 0,
                             float(tx), float(ty), float(tz)))
            elif line.strip() and not line.startswith("|"):
                in_pose_table = False
    return rows


def main():
    if not os.path.isdir(MATERIALS):
        print("找不到素材目录:", MATERIALS, file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    total = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        f.write("# 装甲板真值（相机光学系，单位 m；参考点 = 装甲板几何中心）\n")
        f.write("# 由 scripts/parse_truth_md.py 从 图N.md 的 1.1 表自动生成，勿手改\n")
        f.write("# 列：image,armor,visible,tx,ty,tz\n")
        w = csv.writer(f)
        for md, image in MD_TO_IMAGE.items():
            md_path = os.path.join(MATERIALS, md, f"{md}.md")
            if not os.path.exists(md_path):
                print("跳过不存在的文件:", md_path, file=sys.stderr)
                continue
            rows = parse_md(md_path)
            if not rows:
                print("没解析到内容:", md_path, file=sys.stderr)
                continue
            f.write(f"# {md} -> {image}\n")
            for armor, visible, tx, ty, tz in rows:
                w.writerow([image, armor, visible,
                            f"{tx:+.6f}", f"{ty:+.6f}", f"{tz:+.6f}"])
                total += 1
            print(f"{md}: {len(rows)} 块装甲板 -> {image}")

    print(f"共写入 {total} 行 -> {OUT}")
    if "图2" not in MD_TO_IMAGE:
        print("提示: 图2 还没加入映射（素材/真值配对待确认）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
