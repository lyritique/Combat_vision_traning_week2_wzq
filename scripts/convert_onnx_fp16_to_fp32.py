#!/usr/bin/env python3
"""把 ONNX 模型从 FP16 转成 FP32。

背景：OpenCV 4.5.4 的 dnn 不支持 FP16 常量，加载时会在
      onnx_graph_simplifier.cpp 的 getMatFromTensor 里报
      "Unsupported data type: FLOAT16"。
本脚本把 initializer / Constant / Cast 目标 / 图输入输出类型里的
FLOAT16 全部改成 FLOAT，得到一个 OpenCV 能读的 FP32 模型。

用法:
    python3 scripts/convert_onnx_fp16_to_fp32.py <输入.onnx> [输出.onnx]
    默认输出为 <输入名>_fp32.onnx

依赖:
    pip3 install --user onnx numpy
    注意 onnx 要够新（本模型 ir_version=10，需要 onnx >= 1.16；
    Ubuntu 22.04 apt 里的 python3-onnx 1.7.0 太老，不行）。
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def cast_tensor_to_fp32(tensor) -> bool:
    """把一个 TensorProto 从 FLOAT16 就地转成 FLOAT；返回是否转换过。"""
    if tensor.data_type != TensorProto.FLOAT16:
        return False
    arr = numpy_helper.to_array(tensor).astype(np.float32)
    tensor.CopyFrom(numpy_helper.from_array(arr, tensor.name))
    return True


def fix_value_info(value_info) -> bool:
    if not value_info.type.HasField("tensor_type"):
        return False
    tt = value_info.type.tensor_type
    if tt.elem_type == TensorProto.FLOAT16:
        tt.elem_type = TensorProto.FLOAT
        return True
    return False


def remove_noop_casts(model) -> int:
    """删除 Cast 到 FLOAT 的节点（转成 FP32 后它们是空操作）。

    这些节点来自 PyTorch 的 FP16 导出（输入 FP32 -> cast FP16，输出再 cast 回 FP32）。
    转成 FP32 后它们变成 FLOAT->FLOAT，OpenCV 的 ONNX importer 不会为它们建层，
    下游节点于是拿不到输入（典型报错：Conv 的 inputs.size() != 0）。
    删掉并把下游输入重接到 Cast 的输入即可。
    """
    graph = model.graph
    rename = {}
    keep = []
    for node in graph.node:
        if node.op_type == "Cast":
            to = [a.i for a in node.attribute if a.name == "to"]
            if to and to[0] == TensorProto.FLOAT:
                rename[node.output[0]] = node.input[0]
                continue
        keep.append(node)

    if not rename:
        return 0

    def resolve(name):
        seen = set()
        while name in rename and name not in seen:
            seen.add(name)
            name = rename[name]
        return name

    graph.ClearField("node")
    graph.node.extend(keep)

    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = resolve(inp)

    for out in graph.output:
        if out.name in rename:
            out.name = resolve(out.name)

    # 删掉被移除的中间张量对应的 value_info（不是必须，但更干净）
    keep_vi = [vi for vi in graph.value_info if vi.name not in rename]
    graph.ClearField("value_info")
    graph.value_info.extend(keep_vi)

    return len(rename)


def materialize_broadcasts(model, max_elems: int = 1000000) -> int:
    """把二元算子（Add/Mul/Sub/Div）里的常量预展开成完整形状。

    YOLOv8 的检测头里有这样的写法：
        1x4x2x6300 的张量  Mul  (1x6300) 的常量
    ONNX 的广播规则允许，但 OpenCV 4.5.4 的 Mul 层对高维广播支持不全，
    会在 shape_utils 的 total() 里断言失败。把常量先 broadcast 成完整形状，
    就退化成普通的同形状逐元素运算，OpenCV 能正常处理。
    """
    graph = model.graph
    binary_ops = {"Add", "Sub", "Mul", "Div"}

    # 收集静态形状
    shape = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = [d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim]
        shape[vi.name] = dims

    init = {t.name: t for t in graph.initializer}
    changed = 0

    for node in graph.node:
        if node.op_type not in binary_ops or len(node.input) < 2:
            continue

        a, b = node.input[0], node.input[1]
        const_name = None
        other_shape = None
        if a in init and b in shape:
            const_name, other_shape = a, shape[b]
        elif b in init and a in shape:
            const_name, other_shape = b, shape[a]

        if const_name is None or other_shape is None or 0 in other_shape:
            continue

        arr = numpy_helper.to_array(init[const_name])
        if arr.ndim == 0:            # 标量广播 OpenCV 能处理
            continue

        try:
            target = np.broadcast_shapes(arr.shape, tuple(other_shape))
        except Exception:
            continue

        if tuple(target) == arr.shape:
            continue
        if int(np.prod(target)) > max_elems:
            continue

        expanded = np.broadcast_to(arr, target).astype(np.float32).copy()
        init[const_name].CopyFrom(numpy_helper.from_array(expanded, const_name))
        changed += 1

    return changed


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_fp32.onnx"

    model = onnx.load(src)
    graph = model.graph

    n_init = sum(cast_tensor_to_fp32(t) for t in graph.initializer)

    n_const = 0
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and cast_tensor_to_fp32(attr.t):
                    n_const += 1

    n_cast = 0
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    n_cast += 1

    n_io = sum(fix_value_info(vi)
               for vi in list(graph.input) + list(graph.output) + list(graph.value_info))

    n_noop = remove_noop_casts(model)
    n_bcast = materialize_broadcasts(model)

    try:
        onnx.checker.check_model(model)
        checked = "checker 通过"
    except Exception as exc:          # 老版本 onnx 可能不认识新 opset，只提示不中断
        checked = "checker 警告: " + str(exc)

    onnx.save(model, dst)
    print("转换完成:", src, "->", dst)
    print("  initializer", n_init, "个, Constant", n_const,
          "个, Cast 目标改成 FLOAT", n_cast, "个, 图类型", n_io,
          "个, 删除空 Cast", n_noop, "个, 展开广播常量", n_bcast, "个")
    print(" ", checked)
    print("  输出大小:", os.path.getsize(dst), "字节")
    return 0


if __name__ == "__main__":
    sys.exit(main())
