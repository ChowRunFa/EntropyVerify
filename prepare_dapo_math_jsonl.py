#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 DAPO / DAPO-Math-17k-Processed 数据集，转换为 get_entropy.py 需要的 JSONL 格式。

输入（默认）：
  - data_root: /cephfs/qiuwentao/data
  - repo_id : open-r1/DAPO-Math-17k-Processed 或 BytedTsinghua-SIA/DAPO-Math-17k
  - split   : train

输出（默认）：
  - /cephfs/qiuwentao/data/prepared/<org>/<dataset-name>/<split>_prepared_math.jsonl

每行格式：
  {"text": <题目/完整prompt>, "ground_truth": <标准答案字符串>}
"""

import os
import json
import argparse
from typing import Tuple, Any, Dict

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        "缺少 `datasets` 库，请先安装：\n"
        "  pip install -U datasets\n"
    )


# ---------- 一些路径工具 ----------

def path_from_repo(root: str, repo_id: str) -> str:
    """把 repo_id: 'open-r1/DAPO-Math-17k-Processed' 映射到 root 下的目录。"""
    return os.path.join(root, *repo_id.split("/"))


def infer_snapshot_dir(data_root: str, repo_id: str) -> str:
    """推断下载脚本存放 snapshot 的本地目录。"""
    dataset_dir = path_from_repo(data_root, repo_id)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"找不到数据集目录：{dataset_dir}\n"
            f"请确认你已经用下载脚本把 {repo_id} 下到 {data_root} 下面。"
        )
    return dataset_dir


def default_output_path(out_root: str, repo_id: str, split: str) -> str:
    """
    默认输出路径：
      out_root/prepared/<org>/<dataset-name>/<split>_prepared_math.jsonl
    """
    base_dir = path_from_repo(os.path.join(out_root, "prepared"), repo_id)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{split}_prepared_math.jsonl")


# ---------- 核心抽取逻辑 ----------

def extract_text_and_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    针对一条样本，从不同 schema 中自动抽取：
        text: 题目 / 完整 prompt
        answer: 标准答案（字符串）

    兼容多种常见字段：
      - open-r1 处理版：prompt + solution
      - 原始 DAPO：math_dapo[0].content + MATH['ground_truth']
      - 其它 dataset 变体：text / question / input / instruction + ground_truth
    """
    text = None
    ans = None

    # ----- 抽取 text -----
    # 1) open-r1 的 prompt（通常已经是完整 prompt）
    if isinstance(example.get("prompt"), str) and example["prompt"].strip():
        text = example["prompt"].strip()

    # 2) 原始 DAPO: math_dapo 是一个对话列表，取第一条 content
    if text is None and example.get("math_dapo"):
        v = example["math_dapo"][0]
        if isinstance(v, dict) and "content" in v:
            text = v["content"]

    # 3) 有些版本用 text: list[{"content","role"}] 或 list[str]
    if text is None and example.get("text"):
        v = example["text"][0]
        if isinstance(v, dict) and "content" in v:
            text = v["content"]
        elif isinstance(v, str):
            text = v

    # 4) 再兜底：question / input / instruction 之类
    if text is None:
        for key in ("question", "input", "instruction"):
            if isinstance(example.get(key), str) and example[key].strip():
                text = example[key].strip()
                break

    # ----- 抽取答案 -----
    # 1) open-r1 处理版：solution 字段
    if isinstance(example.get("solution"), str) and example["solution"].strip():
        ans = example["solution"].strip()

    # 2) 独立 ground_truth 字段
    if ans is None and example.get("ground_truth") is not None:
        ans = str(example["ground_truth"]).strip()

    # 3) 原始 DAPO：MATH 是一个 dict，里面有 ground_truth
    if ans is None and isinstance(example.get("MATH"), dict):
        if example["MATH"].get("ground_truth") is not None:
            ans = str(example["MATH"]["ground_truth"]).strip()

    if text is None or ans is None:
        raise ValueError(f"无法从样本中抽取 text/answer，样本 key 列表：{list(example.keys())}")

    return text, ans


# ---------- 主函数 ----------

def main():
    parser = argparse.ArgumentParser(
        description="将 DAPO / DAPO-Math-17k-Processed 转成 JSONL（用于 get_entropy.py）"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/cephfs/qiuwentao/data",
        help="HF 数据集 snapshot 根目录（下载脚本的 local_dir），默认 /cephfs/qiuwentao/data",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="open-r1/DAPO-Math-17k-Processed",
        help=(
            "数据集 repo id：\n"
            "  - open-r1/DAPO-Math-17k-Processed（推荐）\n"
            "  - BytedTsinghua-SIA/DAPO-Math-17k（原始）"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集 split 名称，默认 train",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="/cephfs/qiuwentao/data",
        help="处理后数据统一根目录，默认 /cephfs/qiuwentao/data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "可选：自定义输出路径（绝对路径）。"
            "若不填，则自动生成：<out-root>/prepared/<repo-id>/<split>_prepared_math.jsonl"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="可选：只处理前 N 条样本，默认 None 表示全量",
    )

    args = parser.parse_args()

    # 1) 推断本地 snapshot 目录
    dataset_dir = infer_snapshot_dir(args.data_root, args.repo_id)
    print(f"[INFO] 使用本地数据集目录：{dataset_dir}")
    print(f"[INFO] split = {args.split}")

    # 2) 加载数据
    ds = load_dataset(dataset_dir, split=args.split)
    total = len(ds)
    print(f"[INFO] 加载到样本数：{total}")

    if args.max_samples is not None:
        total = min(total, args.max_samples)
        ds = ds.select(range(total))
        print(f"[INFO] 只处理前 {total} 条样本")

    # 3) 确定输出路径
    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = default_output_path(args.out_root, args.repo_id, args.split)

    print(f"[INFO] 输出到：{out_path}")

    # 4) 逐条写入 JSONL
    n_ok = 0
    n_err = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, ex in enumerate(ds):
            try:
                text, ans = extract_text_and_answer(ex)
            except Exception as e:
                n_err += 1
                # 前几条报一下错误看看 schema 情况
                if n_err <= 5:
                    print(f"[WARN] 第 {i} 条样本抽取失败：{e}")
                continue

            obj = {
                "text": text,
                "ground_truth": ans,
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_ok += 1

            if (i + 1) % 1000 == 0:
                print(f"[INFO] 处理进度：{i+1}/{len(ds)}，成功 {n_ok} 条，失败 {n_err} 条")

    print(f"[DONE] 完成处理，成功 {n_ok} 条，失败 {n_err} 条")
    print(f"[DONE] 最终文件：{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
