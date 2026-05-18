"""
run_all.py
==========
One-command overnight runner: generates AND evaluates all 6 models in sequence.

Usage:
    python run_all.py                         # run every model
    python run_all.py --models graph_rag      # run only one model (for re-runs)
    python run_all.py --skip-generate         # only evaluate (raw files already exist)
    python run_all.py --skip-evaluate         # only generate  (just produce raw JSONL)

Checkpoint-safe: any model whose raw/scored JSONL is already partially written
will resume from where it left off. Safe to Ctrl-C and re-run at any time.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Import your existing system components
# ──────────────────────────────────────────────────────────────────────────────
from rag_system.knowledge_graph import AdvancedKnowledgeGraph
from rag_system.generator import SmartGenerator, BaselineGenerator, NoRetrievalGenerator
from rag_system.logger import Logger
from rag_system.retriever import (VectorBaselineRetriever, LogicGraphRetriever,
                                   QuestionBankRetriever, HybridRetriever)
from rag_system.evaluator import AutomatedEvaluator
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Experiment configuration (matching PPT: 15 topics, 300 questions per model)
# ──────────────────────────────────────────────────────────────────────────────
TEXTBOOK_DIR       = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH      = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"
DATA_DIR           = "experiment_data"

# 15 topics total (matching PPT: 5 formats × 4 questions × 15 topics = 300 per model)
COMPUTATIONAL_TOPICS = [
    "merge sort", "bfs graph traversal", "dfs graph traversal", "dynamic programming knapsack",
    "recursion", "dijkstra shortest path", "binary search",
    "hash table linear probing", "data structures and algorithms"
]

CONCEPTUAL_TOPICS = [
    "cybersecurity", "computer networks", "information retrieval",
    "operating systems", "database systems", "human-computer interaction"
]

TOPICS   = COMPUTATIONAL_TOPICS + CONCEPTUAL_TOPICS
FORMATS  = ["mcq_single", "mcq_multi", "true_false", "fill_blank", "open_answer"]
QUESTIONS_PER_COMBO = 4  # 15 topics × 5 formats × 4 questions = 300 per model

ALL_MODELS = [
    "no_retrieval",
    "vector_rag",
    "hybrid_rag",
    "unpruned_graph_rag",
    "naive_graph_rag",
    "ablation_no_correction",
    "graph_rag",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    """Return HH:MM:SS timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def _md5(d: dict) -> str:
    return hashlib.md5(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _print_banner(text: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {text}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# Shared component initialisation (KG + retrievers)
# ──────────────────────────────────────────────────────────────────────────────

_shared_components: dict = {}   # cache so KG is only loaded once per process


def _get_shared_components(api_key: str):
    """Load KG and retrievers once and reuse across models."""
    global _shared_components
    if _shared_components:
        return _shared_components

    print(f"[{_ts()}] Loading Knowledge Graph (one-time setup) …")
    log_dir = os.path.join(DATA_DIR, "logs_shared")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

    kg = AdvancedKnowledgeGraph(logger=logger)
    if os.path.exists(TEXTBOOK_DIR):
        kg.build_base_structure(TEXTBOOK_DIR)
    else:
        print(f"  [!] TEXTBOOK_DIR '{TEXTBOOK_DIR}' not found — base nodes may be empty.")
    if os.path.exists(TRIPLETS_PATH):
        kg.load_triplets(TRIPLETS_PATH)
    else:
        print(f"  [!] TRIPLETS_PATH '{TRIPLETS_PATH}' not found.")

    from sentence_transformers import SentenceTransformer
    vector_retriever = VectorBaselineRetriever(kg)
    logic_retriever  = LogicGraphRetriever(kg, vector_retriever)
    hybrid_retriever = HybridRetriever(vector_retriever)
    encoder          = SentenceTransformer("all-MiniLM-L6-v2")
    qb_retriever     = QuestionBankRetriever(QUESTION_BANK_PATH, encoder)

    # If textbooks were not available, import question-bank items into
    # the KG so retrievers can still find useful context.
    if not os.path.exists(TEXTBOOK_DIR):
        if os.path.exists(QUESTION_BANK_PATH):
            try:
                kg.load_question_bank(QUESTION_BANK_PATH)
            except Exception as e:
                print(f"  [!] Failed to import question bank into KG: {e}")

    llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    _shared_components = dict(
        kg=kg,
        vector_retriever=vector_retriever,
        logic_retriever=logic_retriever,
        hybrid_retriever=hybrid_retriever,
        qb_retriever=qb_retriever,
        llm_client=llm_client,
    )
    print(f"[{_ts()}] Knowledge Graph loaded successfully.\n")
    return _shared_components


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Generation
# ──────────────────────────────────────────────────────────────────────────────

def run_generation(model_name: str, api_key: str) -> str:
    """
    Generate questions for `model_name`.
    Returns path to the raw JSONL file.
    """
    _print_banner(f"PHASE 1 — GENERATE  |  model: {model_name}  |  {_ts()}")
    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, f"raw_{model_name}.jsonl")

    comp = _get_shared_components(api_key)
    vector_retriever  = comp["vector_retriever"]
    logic_retriever   = comp["logic_retriever"]
    hybrid_retriever  = comp["hybrid_retriever"]
    qb_retriever      = comp["qb_retriever"]

    # Initialise the correct generator
    if model_name == "no_retrieval":
        generator = NoRetrievalGenerator(api_key=api_key)
    elif model_name in ("vector_rag", "hybrid_rag"):
        generator = BaselineGenerator(api_key=api_key)
    else:  # graph_rag, naive_graph_rag, unpruned_graph_rag, ablation_no_correction
        generator = SmartGenerator(api_key=api_key)

    # ── Checkpoint: count already-generated questions ─────────────────────
    completed_counts = {(t, f): 0 for t in TOPICS for f in FORMATS}
    if os.path.exists(output_file):
        print(f"[{_ts()}] Existing raw file found — scanning checkpoints …")
        with open(output_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    key = (d.get("topic"), d.get("question_format"))
                    if key in completed_counts:
                        completed_counts[key] += 1
                except json.JSONDecodeError:
                    continue
        done_so_far = sum(completed_counts.values())
        print(f"[{_ts()}] Resuming — {done_so_far} questions already generated.")

    total_tasks  = len(TOPICS) * len(FORMATS) * QUESTIONS_PER_COMBO
    current_done = sum(completed_counts.values())

    if current_done >= total_tasks:
        print(f"[{_ts()}] Generation for {model_name} already 100% complete — skipping.")
        return output_file

    # ── Main generation loop ──────────────────────────────────────────────
    with open(output_file, "a", encoding="utf-8") as out_f:
        pbar = tqdm(total=total_tasks, initial=current_done,
                    desc=f"  Generating {model_name}", unit="q")

        for topic in TOPICS:
            q_type = "computational" if topic in COMPUTATIONAL_TOPICS else "conceptual"
            for q_format in FORMATS:
                needed = QUESTIONS_PER_COMBO - completed_counts[(topic, q_format)]
                for _ in range(needed):
                    t0 = time.time()
                    try:
                        saved_context = []

                        if model_name == "no_retrieval":
                            raw_json, method = generator.generate(
                                topic=topic,
                                qb_retriever=qb_retriever,
                                question_format=q_format,
                            )

                        elif model_name == "vector_rag":
                            ctx = vector_retriever.retrieve(topic, top_k=5)
                            saved_context = ctx
                            raw_json, method = generator.generate(
                                topic=topic, context=ctx,
                                qb_retriever=qb_retriever,
                                question_format=q_format,
                            )

                        elif model_name == "hybrid_rag":
                            ctx = hybrid_retriever.retrieve(topic, top_k=5)
                            saved_context = ctx
                            raw_json, method = generator.generate(
                                topic=topic, context=ctx,
                                qb_retriever=qb_retriever,
                                question_format=q_format,
                            )

                        else:  # graph_rag / naive_graph_rag / unpruned_graph_rag / ablation_no_correction
                            threshold = -1.0 if model_name == "unpruned_graph_rag" else 0.25

                            graph_ctx = logic_retriever.retrieve_subgraph(
                                topic, hops=2,
                                similarity_threshold=threshold,
                                question_type=q_type,
                            )
                            saved_context = graph_ctx
                            is_naive = (model_name == "naive_graph_rag")
                            skip_correction = (model_name == "ablation_no_correction")
                            raw_json, method = generator.generate(
                                topic=topic, graph_context=graph_ctx,
                                qb_retriever=qb_retriever,
                                question_format=q_format,
                                naive_mode=is_naive,
                                no_self_correction=skip_correction,
                            )

                        q_data = json.loads(raw_json)
                        q_data["topic"]             = topic
                        q_data["model"]             = model_name
                        q_data["method"]            = method
                        q_data["generation_time"]   = round(time.time() - t0, 2)
                        q_data["question_format"]   = q_format
                        q_data["expected_type"]     = q_type
                        q_data["retrieved_context"] = saved_context

                        out_f.write(json.dumps(q_data, ensure_ascii=False) + "\n")
                        out_f.flush()
                        pbar.update(1)

                    except Exception as e:
                        print(f"\n  [!] Gen error — topic={topic} fmt={q_format}: {e}")
                        time.sleep(5)

        pbar.close()

    print(f"[{_ts()}] Generation complete → {output_file}")
    return output_file


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_name: str, raw_file: str, api_key: str) -> str:
    """
    Evaluate the raw JSONL produced by run_generation.
    Returns path to the scored JSONL file.
    """
    _print_banner(f"PHASE 2 — EVALUATE  |  model: {model_name}  |  {_ts()}")
    scored_file = os.path.join(DATA_DIR, f"scored_{model_name}.jsonl")

    comp       = _get_shared_components(api_key)
    llm_client = comp["llm_client"]
    evaluator  = AutomatedEvaluator(llm_client)

    # ── Checkpoint: reload previously scored hashes + restore session memory ──
    scored_hashes: set = set()
    if os.path.exists(scored_file):
        print(f"[{_ts()}] Existing scored file found — restoring checkpoints …")
        with open(scored_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    scored_hashes.add(_md5(d.get("question", {})))
                    # Restore Novelty Penalty memory
                    meta = d.get("metadata", {})
                    key  = (meta.get("topic","").strip().lower(),
                            meta.get("question_format","mcq_single"))
                    evaluator._session_type_counts[key] += 1
                except json.JSONDecodeError:
                    continue
        print(f"[{_ts()}] Skipping {len(scored_hashes)} already-scored questions.")

    # ── Load pending tasks ────────────────────────────────────────────────
    pending = []
    with open(raw_file, encoding="utf-8") as fh:
        for line in fh:
            try:
                item = json.loads(line)
                if _md5(item) not in scored_hashes:
                    pending.append(item)
            except json.JSONDecodeError:
                continue

    if not pending:
        print(f"[{_ts()}] All questions already evaluated — skipping.")
        return scored_file

    print(f"[{_ts()}] {len(pending)} questions to evaluate …")

    # ── Evaluation loop ───────────────────────────────────────────────────
    with open(scored_file, "a", encoding="utf-8") as out_f:
        for item in tqdm(pending, desc=f"  Evaluating {model_name}", unit="q"):
            topic   = item.get("topic", "Unknown")
            q_fmt   = item.get("question_format", "Unknown")
            ctx     = item.get("retrieved_context", [])

            try:
                score = evaluator.evaluate(
                    question_json=json.dumps(item, ensure_ascii=False),
                    context=ctx,
                    topic=topic,
                )
                record = {
                    "metadata": {
                        "topic":           topic,
                        "model":           item.get("model", "unknown"),
                        "method":          item.get("method", "unknown"),
                        "question_format": q_fmt,
                        "expected_type":   item.get("expected_type", "unknown"),
                        "generation_time": item.get("generation_time", 0),
                    },
                    "question": item,
                    "score":    score,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"\n  [!] Eval error — topic={topic}: {e}")
                continue

    print(f"[{_ts()}] Evaluation complete → {scored_file}")
    return scored_file


# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(models: list):
    """Print a comparison table from all scored JSONL files."""
    _print_banner(f"FINAL RESULTS SUMMARY  |  {_ts()}")

    rows = []
    for model in models:
        path = os.path.join(DATA_DIR, f"scored_{model}.jsonl")
        if not os.path.exists(path):
            continue
        scores = []
        fmt_scores: dict = {}
        for line in open(path, encoding="utf-8"):
            try:
                r   = json.loads(line)
                ov  = r["score"]["overall"]
                fmt = r["metadata"]["question_format"]
                scores.append(ov)
                fmt_scores.setdefault(fmt, []).append(ov)
            except Exception:
                continue
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        fb   = sum(fmt_scores.get("fill_blank", [0])) / max(len(fmt_scores.get("fill_blank", [1])), 1)
        rows.append((model, len(scores), mean, fb))

    print(f"\n  {'Model':<28} {'N':>4}  {'Overall':>8}  {'FillBlank':>9}")
    print(f"  {'-'*28} {'-'*4}  {'-'*8}  {'-'*9}")
    for model, n, mean, fb in sorted(rows, key=lambda x: -x[2]):
        print(f"  {model:<28} {n:>4}  {mean:>8.4f}  {fb:>9.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overnight runner: generate + evaluate all models in one command."
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Which models to run (default: all).",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip generation phase (raw JSONL must already exist).",
    )
    parser.add_argument(
        "--skip-evaluate", action="store_true",
        help="Skip evaluation phase (only produce raw JSONL).",
    )
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not found in environment. Check your .env file.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    wall_start = time.time()
    _print_banner(
        f"SmartQG Overnight Runner — {len(args.models)} models  |  start {_ts()}"
    )
    print(f"  Models : {args.models}")
    print(f"  Generate: {'SKIP' if args.skip_generate else 'YES'}")
    print(f"  Evaluate: {'SKIP' if args.skip_evaluate else 'YES'}")
    print()

    failed_models = []

    for model in args.models:
        model_start = time.time()
        try:
            # ── Phase 1: Generate ─────────────────────────────────────────
            raw_file = os.path.join(DATA_DIR, f"raw_{model}.jsonl")
            if not args.skip_generate:
                raw_file = run_generation(model, api_key)
            else:
                if not os.path.exists(raw_file):
                    print(f"  [!] --skip-generate set but {raw_file} not found. Skipping {model}.")
                    failed_models.append(model)
                    continue
                print(f"[{_ts()}] Skipping generation for {model} (--skip-generate).")

            # ── Phase 2: Evaluate ─────────────────────────────────────────
            if not args.skip_evaluate:
                run_evaluation(model, raw_file, api_key)
            else:
                print(f"[{_ts()}] Skipping evaluation for {model} (--skip-evaluate).")

            elapsed = (time.time() - model_start) / 60
            print(f"\n  ✓ {model} finished in {elapsed:.1f} min\n")

        except KeyboardInterrupt:
            print(f"\n[!] Interrupted by user after model '{model}'. Progress is saved.")
            print("[*] Re-run the same command to resume from the checkpoint.")
            break
        except Exception as e:
            print(f"\n[!] Unexpected error in model '{model}':")
            traceback.print_exc()
            failed_models.append(model)
            print("[*] Continuing with next model …\n")
            continue

    # ── Final summary ─────────────────────────────────────────────────────
    if not args.skip_evaluate:
        print_summary(args.models)

    wall_elapsed = (time.time() - wall_start) / 3600
    _print_banner(f"ALL DONE  |  total wall time: {wall_elapsed:.2f} h  |  {_ts()}")
    if failed_models:
        print(f"  [!] These models had errors and may be incomplete: {failed_models}")
        print(f"  [*] Re-run:  python run_all.py --models {' '.join(failed_models)}")
    else:
        print("  All models completed successfully.")
    print()


if __name__ == "__main__":
    main()