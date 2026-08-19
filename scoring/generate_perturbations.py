#!/usr/bin/env python3
# Generate synonym, random-word, and sentence-permutation perturbations for passage scoring

import argparse
import json
import logging
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import spacy
from nltk.corpus import wordnet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNONYM_RATE = 0.30   # fraction of eligible tokens to attempt (synonym)
RANDOM_RATE  = 0.15   # fraction of eligible tokens to attempt (random)
N_RUNS_DEFAULT = 5

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

LIGHT_VERBS = frozenset({
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "make", "makes", "made", "making",
    "get", "gets", "got", "getting", "gotten",
    "go", "goes", "went", "going", "gone",
    "take", "takes", "took", "taking", "taken",
})

POLYSEMY_BLOCKLIST = frozenset({
    "mean", "means", "lead", "leads", "run", "runs", "running",
    "set", "sets", "setting", "light", "lights", "fine", "right",
    "just", "kind", "like", "well", "still", "even", "long",
    "case", "point", "part", "place", "state", "states",
    "form", "forms", "order", "class", "figure", "type",
    "play", "plays", "act", "acts", "fall", "falls",
    "bear", "bears", "spring", "springs", "leaves",
    "bank", "banks", "match", "note", "notes",
    "current", "present", "subject", "object", "power",
    "base", "top", "head", "body", "face", "side",
    "born", "drink", "drinks",
})

# ---------------------------------------------------------------------------
# WordNet helpers
# ---------------------------------------------------------------------------

def ptb_to_wordnet(tag: str):
    if tag.startswith("NN"): return wordnet.NOUN
    if tag.startswith("VB"): return wordnet.VERB
    if tag.startswith("JJ"): return wordnet.ADJ
    if tag.startswith("RB"): return wordnet.ADV
    return None


def precompute_synonyms(word: str, wn_pos, lemma: str) -> List[str]:
    """
    Precompute all usable synonyms from the FIRST WordNet synset.
    Returns a list of candidate synonym strings (may be empty).
    Called once per unique (word, wn_pos) during analysis; results cached.
    """
    if wn_pos is None or word.lower() in POLYSEMY_BLOCKLIST:
        return []

    synsets = wordnet.synsets(word.lower(), pos=wn_pos)
    if not synsets:
        return []

    first_synset = synsets[0]
    word_lower = word.lower()
    lemma_lower = lemma.lower() if lemma else word_lower

    seen = set()
    candidates = []
    for lem in first_synset.lemmas():
        name = lem.name()
        if "_" in name:
            continue
        nl = name.lower()
        if nl == word_lower or nl == lemma_lower:
            continue
        if nl not in seen:
            seen.add(nl)
            candidates.append(name)

    return candidates


def apply_case(original: str, synonym: str) -> str:
    """Match the capitalization of synonym to original word."""
    if original.isupper() and len(original) > 1:
        return synonym.upper()
    elif original[0].isupper():
        return synonym[0].upper() + synonym[1:]
    return synonym.lower()

# ---------------------------------------------------------------------------
# Passage analysis (one spaCy pass, caches everything needed for all runs)
# ---------------------------------------------------------------------------

def has_internal_caps(text: str) -> bool:
    return len(text) > 1 and any(c.isupper() for c in text[1:])


def analyze_passage(doc) -> dict:
    """
    Extract all perturbation-relevant information from a spaCy Doc.
    WordNet synonym candidates are precomputed here so per-run loops
    require no WordNet lookups.
    """
    # Named entity indices
    entity_indices = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            entity_indices.add(i)

    eligible = []
    for i, tok in enumerate(doc):
        if tok.pos_ not in CONTENT_POS:
            continue
        if re.search(r"\d", tok.text):
            continue
        if len(tok.text) < 3:
            continue
        if tok.text.isupper() and len(tok.text) > 1:
            continue
        if has_internal_caps(tok.text):
            continue
        if i in entity_indices:
            continue
        if tok.text.lower() in LIGHT_VERBS:
            continue

        wn_pos = ptb_to_wordnet(tok.tag_)

        # Precompute synonyms now -- avoids repeated WordNet calls in run loop
        synonyms = precompute_synonyms(tok.text, wn_pos, tok.lemma_)

        eligible.append({
            "char_start": tok.idx,
            "char_end":   tok.idx + len(tok.text),
            "text":       tok.text,
            "synonyms":   synonyms,   # precomputed list, may be empty
        })

    # Sentence character spans for B1b
    sentence_spans = [(s.start_char, s.end_char) for s in doc.sents]

    return {
        "eligible":       eligible,
        "sentence_spans": sentence_spans,
    }

# ---------------------------------------------------------------------------
# Perturbation functions (no WordNet calls -- use precomputed synonyms)
# ---------------------------------------------------------------------------

def apply_replacements(text: str, replacements: list) -> str:
    """Apply (char_start, char_end, new_text) replacements right-to-left."""
    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, new in replacements:
        text = text[:start] + new + text[end:]
    return text


def perturb_synonym(text: str, analysis: dict, seed: int) -> str:
    rng = random.Random(seed)
    eligible = analysis["eligible"]
    if not eligible:
        return text

    n_attempt = max(1, round(SYNONYM_RATE * len(eligible)))
    selected = rng.sample(eligible, min(n_attempt, len(eligible)))

    replacements = []
    for tok in selected:
        candidates = tok["synonyms"]
        if not candidates:
            continue
        chosen = apply_case(tok["text"], rng.choice(candidates))
        replacements.append((tok["char_start"], tok["char_end"], chosen))

    return apply_replacements(text, replacements)


def perturb_random(text: str, analysis: dict, seed: int, vocab: List[str]) -> str:
    rng = random.Random(seed)
    eligible = analysis["eligible"]
    if not eligible:
        return text

    n_attempt = max(1, round(RANDOM_RATE * len(eligible)))
    selected = rng.sample(eligible, min(n_attempt, len(eligible)))

    replacements = []
    for tok in selected:
        word = tok["text"]
        for _ in range(10):
            repl = rng.choice(vocab)
            if repl.lower() != word.lower():
                break
        else:
            continue
        replacements.append((
            tok["char_start"], tok["char_end"],
            apply_case(word, repl),
        ))

    return apply_replacements(text, replacements)


def perturb_sentence(text: str, analysis: dict, seed: int) -> str:
    spans = analysis["sentence_spans"]
    if len(spans) <= 1:
        return text

    rng = random.Random(seed)
    sentences = [text[s:e] for s, e in spans]
    separators = [text[spans[i][1]:spans[i+1][0]] for i in range(len(spans)-1)]

    original = list(range(len(spans)))
    order = list(range(len(spans)))
    for _ in range(20):
        rng.shuffle(order)
        if order != original:
            break
    else:
        order = list(reversed(original))

    prefix = text[:spans[0][0]]
    suffix = text[spans[-1][1]:]
    parts = [prefix]
    for i, idx in enumerate(order):
        parts.append(sentences[idx])
        if i < len(spans) - 1:
            parts.append(separators[i] if i < len(separators) else " ")
    parts.append(suffix)
    return "".join(parts)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_passages(paths: List[str]) -> Dict[str, str]:
    passages = {}
    for path in paths:
        n_before = len(passages)
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pid = str(obj["pid"])
                if pid not in passages:
                    passages[pid] = obj["passage"]
        logging.info("Loaded %d new passages from %s (total: %d)",
                     len(passages) - n_before, path, len(passages))
    return passages

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate B1 passage perturbations (local, CPU only)."
    )
    parser.add_argument("--passages", required=True, nargs="+",
                        help="Path(s) to judged_passages.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for perturbed passage files")
    parser.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT)
    parser.add_argument("--operators", nargs="+",
                        default=["synonym", "random", "sentence"],
                        choices=["synonym", "random", "sentence"])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t_total = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load passages
    # ------------------------------------------------------------------ #
    passages = load_passages(args.passages)
    pids = sorted(passages.keys())
    logging.info("Total passages: %d", len(pids))

    # ------------------------------------------------------------------ #
    # 2. spaCy analysis + WordNet precomputation (one pass)
    # ------------------------------------------------------------------ #
    logging.info("Loading spaCy en_core_web_md ...")
    nlp = spacy.load("en_core_web_md")
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    logging.info("Analyzing %d passages (spaCy POS+NER + WordNet precomputation) ...",
                 len(pids))
    logging.info("This is the slow step. Progress printed every 5,000 passages.")
    t_analyze = time.time()

    analyses: Dict[str, dict] = {}
    texts = [passages[pid] for pid in pids]

    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        analyses[pids[i]] = analyze_passage(doc)
        n_done = i + 1
        if n_done % 5000 == 0 or n_done == len(pids):
            elapsed = time.time() - t_analyze
            rate = n_done / elapsed
            remaining = (len(pids) - n_done) / rate if rate > 0 else 0
            logging.info("  %d / %d passages analyzed (%.0f/sec, ~%.1f min remaining)",
                         n_done, len(pids), rate, remaining / 60)

    elapsed_analyze = time.time() - t_analyze
    logging.info("Analysis complete in %.1f min", elapsed_analyze / 60)

    # Quick stats
    n_zero = sum(1 for a in analyses.values() if not a["eligible"])
    n_single = sum(1 for a in analyses.values() if len(a["sentence_spans"]) <= 1)
    mean_elig = sum(len(a["eligible"]) for a in analyses.values()) / len(analyses)
    n_syn_eligible = sum(
        1 for a in analyses.values()
        for t in a["eligible"] if t["synonyms"]
    )
    logging.info("  Passages with 0 eligible tokens:   %d (%.1f%%)",
                 n_zero, 100 * n_zero / len(pids))
    logging.info("  Passages with <= 1 sentence:        %d (%.1f%%)",
                 n_single, 100 * n_single / len(pids))
    logging.info("  Mean eligible tokens per passage:   %.1f", mean_elig)
    logging.info("  Eligible tokens with synonyms:      %d", n_syn_eligible)

    # ------------------------------------------------------------------ #
    # 3. Build vocabulary pool for random replacement
    # ------------------------------------------------------------------ #
    vocab = sorted({
        tok["text"].lower()
        for a in analyses.values()
        for tok in a["eligible"]
    })
    logging.info("Vocabulary pool for random replacement: %d unique words", len(vocab))

    # ------------------------------------------------------------------ #
    # 4. Generate perturbed passage files
    # ------------------------------------------------------------------ #
    op_config = {
        "synonym":  ("b1a_syn",  perturb_synonym),
        "random":   ("b1a_rand", perturb_random),
        "sentence": ("b1b_sent", perturb_sentence),
    }

    for op_name in args.operators:
        prefix, perturb_fn = op_config[op_name]
        logging.info("=" * 60)
        logging.info("Operator: %s  |  prefix: %s  |  %d runs",
                     op_name, prefix, args.n_runs)

        for run_idx in range(args.n_runs):
            out_path  = os.path.join(args.output_dir, f"{prefix}_run{run_idx}.jsonl")
            meta_path = os.path.join(args.output_dir, f"{prefix}_run{run_idx}_meta.json")

            if os.path.exists(out_path):
                logging.info("  run %d: already exists, skipping -> %s",
                             run_idx, out_path)
                continue

            # Derive unique per-passage seed from run index
            run_rng = random.Random(run_idx)
            passage_seeds = {pid: run_rng.randint(0, 2**31) for pid in pids}

            t_run = time.time()
            n_perturbed = 0
            n_subs_total = 0

            with open(out_path, "w", encoding="utf-8") as f:
                for j, pid in enumerate(pids):
                    text     = passages[pid]
                    analysis = analyses[pid]
                    pseed    = passage_seeds[pid]

                    if op_name == "synonym":
                        perturbed = perturb_synonym(text, analysis, pseed)
                    elif op_name == "random":
                        perturbed = perturb_random(text, analysis, pseed, vocab)
                    elif op_name == "sentence":
                        perturbed = perturb_sentence(text, analysis, pseed)

                    f.write(json.dumps({"pid": pid, "passage": perturbed}) + "\n")

                    if perturbed != text:
                        n_perturbed += 1

                    # Progress every 20,000 passages within a run
                    if (j + 1) % 20000 == 0:
                        logging.info("    run %d: %d / %d passages written ...",
                                     run_idx, j + 1, len(pids))

            elapsed_run = time.time() - t_run
            size_mb = os.path.getsize(out_path) / (1024 * 1024)

            # Save metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "operator": op_name, "prefix": prefix,
                    "run_index": run_idx, "seed": run_idx,
                    "rate": SYNONYM_RATE if op_name == "synonym"
                            else RANDOM_RATE if op_name == "random"
                            else None,
                    "n_passages": len(pids),
                    "n_perturbed": n_perturbed,
                    "pct_perturbed": round(100 * n_perturbed / len(pids), 1),
                    "elapsed_seconds": round(elapsed_run, 1),
                }, f, indent=2)

            logging.info(
                "  run %d: %d/%d passages changed (%.1f%%) in %.1f sec -> %s (%.1f MB)",
                run_idx, n_perturbed, len(pids),
                100 * n_perturbed / len(pids),
                elapsed_run, out_path, size_mb,
            )

    # ------------------------------------------------------------------ #
    # 5. Final summary
    # ------------------------------------------------------------------ #
    elapsed_total = time.time() - t_total
    logging.info("=" * 60)
    logging.info("All done in %.1f min", elapsed_total / 60)
    logging.info("Files in %s:", args.output_dir)
    for fname in sorted(os.listdir(args.output_dir)):
        if fname.endswith(".jsonl"):
            size_mb = os.path.getsize(
                os.path.join(args.output_dir, fname)
            ) / (1024 * 1024)
            logging.info("  %-35s %.1f MB", fname, size_mb)
    logging.info("")
    logging.info("Next: upload perturbed_passages/ to RunPod and run: bash run_b1.sh")


if __name__ == "__main__":
    main()
