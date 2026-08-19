#!/usr/bin/env python3
# Score passages under three instruction-prompt variants for stability measurement

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

# Variant 0: Original (UMBRELA-style), identical to score_passages.py
SYSTEM_PROMPT_V0 = (
    "You are a search quality rater evaluating the relevance of web passages. "
    "Given a query and a passage, you must provide a score on an integer scale "
    "of 0 to 3 with the following meanings:\n\n"
    "0: The passage is not relevant to the query.\n"
    "1: The passage is related to the query but does not directly answer it.\n"
    "2: The passage has some answer for the query, but the answer may be a bit "
    "unclear, or hidden amongst extraneous information.\n"
    "3: The passage is dedicated to the query and contains the exact answer."
)
USER_TEMPLATE_V0 = "Query: {query}\nPassage: {passage}\n\nScore:"

# Variant 1: Rubric paraphrase
# Same four-level meaning, entirely different wording throughout.
# Grounded in: IRT reliability paper (arXiv 2602.00521), which applies
# synonym-level paraphrase to the instruction as a semantics-preserving
# perturbation and treats score fluctuation as a reliability signal.
SYSTEM_PROMPT_V1 = (
    "You are an information retrieval assessor judging how well a web passage "
    "addresses a search query. Rate the relevance of the passage on a scale "
    "from 0 to 3 using the criteria below:\n\n"
    "0: The passage is entirely off-topic and does not address the query.\n"
    "1: The passage is somewhat related to the query but fails to provide "
    "a direct answer.\n"
    "2: The passage partially answers the query, though the answer may be "
    "buried in irrelevant details or lack clarity.\n"
    "3: The passage fully and directly answers the query with dedicated, "
    "precise information."
)
USER_TEMPLATE_V1 = "Search query: {query}\nWeb passage: {passage}\n\nRelevance score:"

# Variant 2: Descending rubric order
# Same wording as original but rubric levels presented 3 to 0 instead of 0 to 3.
# Grounded in: Scoring bias paper (arXiv 2506.22316), which finds that
# descending rubric order produces measurable scoring shifts and recommends
# testing it as a robustness check. This is their strongest perturbation
# finding for pointwise judges.
SYSTEM_PROMPT_V2 = (
    "You are a search quality rater evaluating the relevance of web passages. "
    "Given a query and a passage, you must provide a score on an integer scale "
    "of 0 to 3 with the following meanings:\n\n"
    "3: The passage is dedicated to the query and contains the exact answer.\n"
    "2: The passage has some answer for the query, but the answer may be a bit "
    "unclear, or hidden amongst extraneous information.\n"
    "1: The passage is related to the query but does not directly answer it.\n"
    "0: The passage is not relevant to the query."
)
USER_TEMPLATE_V2 = "Query: {query}\nPassage: {passage}\n\nScore:"

# Variant 3: Minimal surface change (formatting and label style only)
# Same rubric content and order, but different label formatting.
# Grounded in: Scoring bias paper's score-ID axis (arXiv 2506.22316),
# which tests numeric vs letter vs Roman numeral score identifiers and
# finds even this minimal change shifts scores. We adapt this to our
# template by changing the label presentation.
SYSTEM_PROMPT_V3 = (
    "You are a search quality rater evaluating the relevance of web passages. "
    "Given a query and a passage, you must provide a score on an integer scale "
    "of 0 to 3 with the following meanings:\n\n"
    "Score 0 - The passage is not relevant to the query.\n"
    "Score 1 - The passage is related to the query but does not directly answer it.\n"
    "Score 2 - The passage has some answer for the query, but the answer may be a bit "
    "unclear, or hidden amongst extraneous information.\n"
    "Score 3 - The passage is dedicated to the query and contains the exact answer."
)
USER_TEMPLATE_V3 = "Query text: {query}\nText passage: {passage}\n\nYour score:"

VARIANTS = {
    0: (SYSTEM_PROMPT_V0, USER_TEMPLATE_V0),
    1: (SYSTEM_PROMPT_V1, USER_TEMPLATE_V1),
    2: (SYSTEM_PROMPT_V2, USER_TEMPLATE_V2),
    3: (SYSTEM_PROMPT_V3, USER_TEMPLATE_V3),
}

VARIANT_DESCRIPTIONS = {
    0: "Original (UMBRELA-style, ascending 0-3)",
    1: "Rubric paraphrase (different wording, same meaning)",
    2: "Descending rubric order (3-0 instead of 0-3)",
    3: "Minimal surface change (label formatting only)",
}


# ---------------------------------------------------------------------------
# Data loading (identical to score_passages.py)
# ---------------------------------------------------------------------------

def load_queries(paths: List[str]) -> Dict[str, str]:
    """Load queries from one or more TSV files (qid<TAB>query)."""
    queries = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
        logging.info("Loaded %d queries total (including %s)", len(queries), path)
    return queries


def load_qrels(paths: List[str]) -> Dict[str, Dict[str, int]]:
    """Load qrels from one or more TREC-format files."""
    qrels: Dict[str, Dict[str, int]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, pid, rel = parts[0], parts[2], int(parts[3])
                    qrels.setdefault(qid, {})[pid] = rel
    total_pairs = sum(len(v) for v in qrels.values())
    logging.info("Loaded qrels: %d queries, %d judged pairs", len(qrels), total_pairs)
    return qrels


def load_passages(path: str) -> Dict[str, str]:
    """Load passages from JSONL file."""
    passages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passages[str(obj["pid"])] = obj["passage"]
    logging.info("Loaded %d passages from %s", len(passages), path)
    return passages


# ---------------------------------------------------------------------------
# Token handling (identical to score_passages.py)
# ---------------------------------------------------------------------------

def resolve_score_token_ids(tokenizer) -> Tuple[Dict[int, int], List[int]]:
    """Identify token IDs for score digits 0-3."""
    token_to_score: Dict[int, int] = {}
    for score in range(4):
        for variant in [str(score), f" {score}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_to_score[ids[0]] = score

    scores_found = set(token_to_score.values())
    if scores_found != {0, 1, 2, 3}:
        missing = {0, 1, 2, 3} - scores_found
        raise RuntimeError(f"Could not find token IDs for scores: {missing}")

    allowed_ids = sorted(token_to_score.keys())
    logging.info(
        "Score token mapping: %s",
        {v: k for k, v in sorted(token_to_score.items(), key=lambda x: x[1])},
    )
    return token_to_score, allowed_ids


def build_logits_processor(allowed_ids: List[int]):
    """Mask all tokens except allowed score tokens."""
    def processor(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(logits, float("-inf"))
        for tid in allowed_ids:
            mask[tid] = 0.0
        return logits + mask
    return processor


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_chat_prompt(
    tokenizer,
    query: str,
    passage: str,
    system_prompt: str,
    user_template: str,
    max_passage_tokens: int = 512,
) -> str:
    """Build a chat-formatted prompt using the specified variant."""
    passage_ids = tokenizer.encode(passage, add_special_tokens=False)
    if len(passage_ids) > max_passage_tokens:
        passage = tokenizer.decode(
            passage_ids[:max_passage_tokens], skip_special_tokens=True
        )

    user_content = user_template.format(query=query, passage=passage)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_query(
    llm: LLM,
    tokenizer,
    query_id: str,
    query_text: str,
    passage_ids: List[str],
    passages: Dict[str, str],
    token_to_score: Dict[int, int],
    allowed_ids: List[int],
    system_prompt: str,
    user_template: str,
    max_passage_tokens: int = 512,
) -> List[dict]:
    """Score all judged passages for a single query using the given prompt variant."""
    prompts = []
    valid_pids = []
    for pid in passage_ids:
        if pid not in passages:
            logging.warning("Query %s: passage %s not found, skipping", query_id, pid)
            continue
        prompt = build_chat_prompt(
            tokenizer, query_text, passages[pid],
            system_prompt, user_template, max_passage_tokens,
        )
        prompts.append(prompt)
        valid_pids.append(pid)

    if not prompts:
        logging.warning("Query %s: no valid passages to score", query_id)
        return []

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,
        logits_processors=[build_logits_processor(allowed_ids)],
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for pid, output in zip(valid_pids, outputs):
        generated = output.outputs[0]
        gen_token_id = generated.token_ids[0]

        if gen_token_id in token_to_score:
            score = token_to_score[gen_token_id]
        else:
            logging.error(
                "Query %s, passage %s: unexpected token_id %d. Falling back to argmax.",
                query_id, pid, gen_token_id,
            )
            score = -1

        token_logprobs = generated.logprobs[0]
        raw_logprobs = {}
        for tid, score_val in token_to_score.items():
            if tid in token_logprobs:
                raw_logprobs[score_val] = token_logprobs[tid].logprob
            else:
                raw_logprobs[score_val] = -100.0

        if score == -1:
            score = max(raw_logprobs, key=raw_logprobs.get)

        max_lp = max(raw_logprobs.values())
        exp_vals = {s: math.exp(lp - max_lp) for s, lp in raw_logprobs.items()}
        total = sum(exp_vals.values())
        probs = {s: exp_vals[s] / total for s in range(4)}

        results.append({
            "query_id": query_id,
            "passage_id": pid,
            "score": score,
            "logprobs": {str(s): round(raw_logprobs[s], 6) for s in range(4)},
            "probs": {str(s): round(probs[s], 6) for s in range(4)},
        })

    return results


# ---------------------------------------------------------------------------
# Checkpointing (identical to score_passages.py)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        completed = set(data.get("completed_queries", []))
        logging.info("Resumed from checkpoint: %d queries already done", len(completed))
        return completed
    return set()


def save_checkpoint(checkpoint_path: str, completed: set):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"completed_queries": sorted(completed)}, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="B3 instruction perturbation scoring."
    )
    parser.add_argument(
        "--queries", required=True, nargs="+",
        help="Path(s) to queries TSV files",
    )
    parser.add_argument(
        "--qrels", required=True, nargs="+",
        help="Path(s) to qrels (TREC format)",
    )
    parser.add_argument("--passages", required=True, help="Path to passages JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument(
        "--variant", required=True, type=int, choices=[0, 1, 2, 3],
        help="Prompt variant: 0=original, 1=paraphrase, 2=descending, 3=minimal",
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--max-passage-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Select prompt variant
    system_prompt, user_template = VARIANTS[args.variant]
    logging.info(
        "Using prompt variant %d: %s", args.variant,
        VARIANT_DESCRIPTIONS[args.variant],
    )
    logging.info("System prompt preview: %s...", system_prompt[:80])

    # Load data
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    passages = load_passages(args.passages)

    query_ids = sorted(set(queries.keys()) & set(qrels.keys()))
    logging.info("Queries to score: %d", len(query_ids))

    total_pairs = sum(len(qrels[qid]) for qid in query_ids)
    logging.info("Total (query, passage) pairs: %d", total_pairs)

    # Check for missing passages
    missing = 0
    for qid in query_ids:
        for pid in qrels[qid]:
            if pid not in passages:
                missing += 1
    if missing > 0:
        logging.warning("%d judged passages not found. They will be skipped.", missing)

    # Checkpointing
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.output + ".checkpoint.json"
    completed = load_checkpoint(checkpoint_path)

    remaining = [qid for qid in query_ids if qid not in completed]
    logging.info("Queries remaining: %d (of %d total)", len(remaining), len(query_ids))

    if not remaining:
        logging.info("All queries already scored. Nothing to do.")
        return

    # Initialize model
    logging.info("Loading model: %s", args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    token_to_score, allowed_ids = resolve_score_token_ids(tokenizer)

    # Scoring loop
    logging.info("Starting B3 scoring run (variant %d)", args.variant)
    t_start = time.time()
    pairs_scored = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, qid in enumerate(remaining):
            query_text = queries[qid]
            passage_ids = sorted(qrels[qid].keys())

            logging.info(
                "Scoring query %d/%d: %s (%d passages)",
                i + 1, len(remaining), qid, len(passage_ids),
            )

            results = score_query(
                llm=llm,
                tokenizer=tokenizer,
                query_id=qid,
                query_text=query_text,
                passage_ids=passage_ids,
                passages=passages,
                token_to_score=token_to_score,
                allowed_ids=allowed_ids,
                system_prompt=system_prompt,
                user_template=user_template,
                max_passage_tokens=args.max_passage_tokens,
            )

            for r in results:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()

            completed.add(qid)
            save_checkpoint(checkpoint_path, completed)
            pairs_scored += len(results)

            elapsed = time.time() - t_start
            rate = pairs_scored / elapsed if elapsed > 0 else 0
            logging.info(
                "  -> %d pairs scored (%.1f pairs/sec, %.1f min elapsed)",
                pairs_scored, rate, elapsed / 60,
            )

    elapsed_total = time.time() - t_start
    logging.info(
        "Done. Variant %d: scored %d pairs across %d queries in %.1f minutes.",
        args.variant, pairs_scored, len(remaining), elapsed_total / 60,
    )
    logging.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
