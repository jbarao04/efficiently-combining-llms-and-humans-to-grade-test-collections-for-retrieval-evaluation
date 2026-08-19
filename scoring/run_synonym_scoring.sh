#!/bin/bash
# ==========================================================================
# Family B1a: Synonym Substitution — Scoring Runs
# ==========================================================================
# Run this on RunPod with an RTX 4090.
# Estimated time: ~1.5 hours (~$1.10)
#
# Structure:
#   1 operator x 3 runs x 2 corpora = 6 runs
# ==========================================================================

set -e

export OPENAI_API_KEY=dummy

mkdir -p results/family_b/b1

echo "============================================"
echo "B1a Synonym Substitution Scoring"
echo "============================================"
echo "1 operator x 3 runs x 2 corpora = 6 runs"
echo "~431,736 total (query, passage) pairs"
echo "Estimated time: ~1.5 hours on RTX 4090"
echo ""

for R in 0 1 2; do
    OUTPUT="results/family_b/b1/b1a_syn_run${R}_v1.jsonl"
    if [ -f "$OUTPUT" ] && [ ! -f "${OUTPUT}.checkpoint.json" ]; then
        echo "SKIP: $OUTPUT (already complete)"
    else
        echo "--- b1a_syn run${R} on v1 ---"
        python score_passages.py --queries data/v1/queries_merged.tsv --qrels data/v1/qrels_merged.txt --passages perturbed_passages/b1a_syn_run${R}.jsonl --output "$OUTPUT"
    fi

    OUTPUT="results/family_b/b1/b1a_syn_run${R}_v2.jsonl"
    if [ -f "$OUTPUT" ] && [ ! -f "${OUTPUT}.checkpoint.json" ]; then
        echo "SKIP: $OUTPUT (already complete)"
    else
        echo "--- b1a_syn run${R} on v2 ---"
        python score_passages.py --queries data/v2/queries_merged.tsv --qrels data/v2/qrels_merged.txt --passages perturbed_passages/b1a_syn_run${R}.jsonl --output "$OUTPUT"
    fi
done

echo ""
echo "============================================"
echo "B1a scoring complete!"
echo "============================================"
echo ""
echo "Output files:"
ls -la results/family_b/b1/b1a_syn_*.jsonl 2>/dev/null
echo ""
echo "Total: $(ls results/family_b/b1/b1a_syn_*.jsonl 2>/dev/null | wc -l) files (expected: 6)"
echo ""
echo "Download results/family_b/b1/ and stop the pod."
