import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ----------------------------------------------------------------------------- #
# 工具：从 top-k logprobs 计算熵
# ----------------------------------------------------------------------------- #
def entropy_from_logits(topk_dict):
    """Calculate entropy from top-k logprobs."""
    probs = [math.exp(lp.logprob) for lp in topk_dict.values()]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def print_and_mark_high_entropy(res, top_percent=0.2):
    """Mark top-percent high-entropy tokens and write back to pos_norm/high_entropy."""
    if not res["token_id"]:
        # 没有 token 的话，就直接返回
        res["pos_norm"] = []
        res["high_entropy"] = []
        return res

    df = pd.DataFrame({
        "token_id": res["token_id"],
        "token":    res["token"],
        "entropy":  res["entropy"],
        "logprob":  res["logprob"],
        "surprise": res["surprise"],
    })
    df["pos_norm"] = np.linspace(0, 1, len(df))
    n = max(1, int(len(df) * top_percent))
    high_idx = set(df.nlargest(n, "entropy").index)
    df["high_entropy"] = df.index.isin(high_idx)
    res["pos_norm"]     = df["pos_norm"].tolist()
    res["high_entropy"] = df["high_entropy"].tolist()
    return res


def extract_answer(text):
    """Extract content after 'Answer' from the text."""
    if "Answer" in text:
        return text.split("Answer")[-1].strip()
    if "answer" in text:
        return text.split("answer")[-1].strip()
    return text.strip()


def judge_answer(answer, ground_truth) -> int:
    """Simple scoring: check if the ground_truth string is contained in the extracted answer."""
    return 1 if str(ground_truth) in extract_answer(answer) else 0


# ----------------------------------------------------------------------------- #
# to be called by `batch_generate_and_entropy`
# ----------------------------------------------------------------------------- #
@torch.inference_mode()
def batch_generate_and_entropy(
    model,
    tokenizer,
    prompts,
    sampling_params,
    start_idx=0,
    # 我们先把 processor 去掉，保证每条都写
    processor_names=None,
):
    # 1) construct prompt
    texts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a smart assistant, good at answering various questions. "
                        "Please use thinking mode when solving the problem."
                    ),
                },
                {"role": "user", "content": p["text"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in prompts
    ]

    # 2) inference
    outputs = model.generate(texts, sampling_params)

    # 3) to list of dict (不做任何过滤，全部保留)
    items = []
    for i, output in enumerate(outputs):
        for j, resp in enumerate(output.outputs):
            items.append(
                {
                    "prompt_idx": i,
                    "sample_idx": j,
                    "resp": resp,
                    "prompt": prompts[i],
                }
            )

    # 4) calculate entropy for **all** items
    results = []
    for it in items:
        resp = it["resp"]
        gen_tokens = resp.token_ids
        logprobs = resp.logprobs
        gen_text = resp.text

        if THINK_ONLY:
            try:
                s = gen_tokens.index(THINK_TOKEN)
                e = gen_tokens.index(END_THINK_TOKEN)
                gen_tokens = gen_tokens[s + 1 : e]
                logprobs = logprobs[s + 1 : e]
            except ValueError:
                gen_tokens, logprobs, gen_text = [], [], ""

        res = {
            "prompt_idx": it["prompt_idx"] + start_idx,
            "sample_idx": it["sample_idx"],
            "prompt_text": it["prompt"]["text"],
            "ground_truth": it["prompt"]["ground_truth"],
            "gen_text": gen_text,
            "token_id": [],
            "token": [],
            "entropy": [],
            "logprob": [],
            "surprise": [],
            "high_entropy": [],
        }

        for tok_id, topk_dict in zip(gen_tokens, logprobs):
            ent = entropy_from_logits(topk_dict)
            # topk_dict 是 {token_id: Logprob} 这样的 dict
            if tok_id in topk_dict:
                lp = topk_dict[tok_id].logprob
            else:
                # 理论上不会太多出现，兜底给个 nan
                any_lp = next(iter(topk_dict.values()))
                lp = any_lp.logprob
            sp = -lp / math.log(2)
            tk = tokenizer.convert_ids_to_tokens([tok_id])[0]

            res["token_id"].append(tok_id)
            res["token"].append(tk)
            res["entropy"].append(ent)
            res["logprob"].append(lp)
            res["surprise"].append(sp)

        res = print_and_mark_high_entropy(res, top_percent=0.2)
        results.append(res)

    return results, items  # 返回 entropy 结果 + 原始 items（含 resp）


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    MODEL_PATH = "/cephfs/qiuwentao/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    DATA_DIR = "/cephfs/qiuwentao/data"
    INPUT_JSONL = (
        "/cephfs/qiuwentao/data/PEA/prepared/open-r1/"
        "DAPO-Math-17k-Processed/train_prepared_math.jsonl"
    )

    # 带 entropy 的主结果
    OUTPUT_JSONL = (
        "/cephfs/qiuwentao/data/PEA/with_entropy/with_entropy_dapo_math.jsonl"
    )
    # 额外增加一个原始输出文件，每条都写，方便你看
    RAW_OUTPUT_JSONL = (
        "/cephfs/qiuwentao/data/PEA/with_entropy/raw_outputs_dapo_math.jsonl"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    BATCH_SIZE = 50
    THINK_ONLY = True
    THINK_TOKEN = 151667
    END_THINK_TOKEN = 151668

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = LLM(MODEL_PATH)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
        logprobs=20,
        prompt_logprobs=None,
        n=8,
    )

    # ★ 先禁用 processor，保证每一条都能进文件
    processor_chain = None
    results_buffer = []
    raw_buffer = []
    batch_prompts = []

    with open(INPUT_JSONL, encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout_entropy, \
         open(RAW_OUTPUT_JSONL, "w", encoding="utf-8") as fout_raw:

        idx = 0
        for line in tqdm(fin):
            obj = json.loads(line)
            batch_prompts.append(
                {
                    "text": obj["text"],
                    "ground_truth": obj["ground_truth"],
                }
            )

            if len(batch_prompts) == BATCH_SIZE:
                batch_res, raw_items = batch_generate_and_entropy(
                    model,
                    tokenizer,
                    batch_prompts,
                    sampling_params,
                    start_idx=idx - BATCH_SIZE + 1,
                    processor_names=processor_chain,
                )
                # 主结果：带 entropy 的
                results_buffer.extend(batch_res)

                # 原始输出：每条 sample 原封不动写进去（只保留 json 可序列化的信息）
                for it in raw_items:
                    fout_raw.write(
                        json.dumps(
                            {
                                "prompt_idx": it["prompt_idx"],
                                "sample_idx": it["sample_idx"],
                                "prompt_text": it["prompt"]["text"],
                                "ground_truth": it["prompt"]["ground_truth"],
                                "gen_text": it["resp"].text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                fout_raw.flush()

                batch_prompts.clear()

            idx += 1

            # 定期把 entropy 结果刷盘
            if len(results_buffer) >= 50:
                for r in results_buffer:
                    fout_entropy.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout_entropy.flush()
                results_buffer.clear()

        # 末尾如果还有残留 batch
        if batch_prompts:
            batch_res, raw_items = batch_generate_and_entropy(
                model,
                tokenizer,
                batch_prompts,
                sampling_params,
                start_idx=idx - len(batch_prompts),
                processor_names=processor_chain,
            )
            results_buffer.extend(batch_res)
            for it in raw_items:
                fout_raw.write(
                    json.dumps(
                        {
                            "prompt_idx": it["prompt_idx"],
                            "sample_idx": it["sample_idx"],
                            "prompt_text": it["prompt"]["text"],
                            "ground_truth": it["prompt"]["ground_truth"],
                            "gen_text": it["resp"].text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fout_raw.flush()

        # 把剩下的 entropy 结果写掉
        for r in results_buffer:
            fout_entropy.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout_entropy.flush()

    print(f"Entropy results written to {OUTPUT_JSONL}")
    print(f"Raw outputs written to {RAW_OUTPUT_JSONL}")
