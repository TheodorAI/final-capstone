"""single_query.py
=================
Run a single-topic / single-query generation and print one JSON result.

Usage examples:
  python single_query.py --topic "hash table linear probing" --generator graph_rag
  python single_query.py --topic "merge sort" --generator vector_rag --format mcq_single

This script mirrors the generator selection used in run_all.py but
operates on a single input and returns a single JSON object.
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

from rag_system.knowledge_graph import AdvancedKnowledgeGraph
from rag_system.generator import NoRetrievalGenerator, BaselineGenerator, SmartGenerator
from rag_system.retriever import VectorBaselineRetriever, LogicGraphRetriever, QuestionBankRetriever
from rag_system.logger import Logger
from sentence_transformers import SentenceTransformer

# Default paths (same as run_all.py)
TEXTBOOK_DIR = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"


def _init_shared(api_key: str):
    out_dir = "single_query_logs"
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(out_dir)

    kg = AdvancedKnowledgeGraph(logger)
    if os.path.exists(TEXTBOOK_DIR):
        kg.build_base_structure(TEXTBOOK_DIR)
    else:
        logger.log(f"  [!] TEXTBOOK_DIR '{TEXTBOOK_DIR}' not found — base nodes may be empty.")

    if os.path.exists(TRIPLETS_PATH):
        kg.load_triplets(TRIPLETS_PATH)
    else:
        logger.log(f"  [!] TRIPLETS_PATH '{TRIPLETS_PATH}' not found.")

    # Retrievers
    vector_retriever = VectorBaselineRetriever(kg)
    logic_retriever = LogicGraphRetriever(kg, vector_retriever)

    # Question bank encoder + retriever (few-shot)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    qb_retriever = QuestionBankRetriever(QUESTION_BANK_PATH, encoder)

    return {
        "logger": logger,
        "kg": kg,
        "vector_retriever": vector_retriever,
        "logic_retriever": logic_retriever,
        "qb_retriever": qb_retriever,
    }


def generate_single(api_key: str, topic: str, generator: str = "graph_rag",
                    question_format: str = "mcq_single", top_k: int = 5, hops: int = 2,
                    comps: dict = None):
    """
    Generate a single question. If `comps` (shared components) is provided,
    use it; otherwise initialise shared components.
    """
    if comps is None:
        comps = _init_shared(api_key)

    qb = comps["qb_retriever"]
    if generator == "no_retrieval":
        gen = NoRetrievalGenerator(api_key=api_key)
        raw_json, method = gen.generate(topic=topic, qb_retriever=qb, question_format=question_format)
        ctx = []

    elif generator == "vector_rag":
        gen = BaselineGenerator(api_key=api_key)
        ctx = comps["vector_retriever"].retrieve(topic, top_k=top_k)
        raw_json, method = gen.generate(topic=topic, context=ctx, qb_retriever=qb, question_format=question_format)

    else:  # graph_rag
        gen = SmartGenerator(api_key=api_key)
        # follow run_all defaults for thresholds
        question_type = "computational" if any(k in topic.lower() for k in ("sort","hash","tree","bfs","dfs","dynamic","knapsack","recursion","dijkstra","binary search","heap")) else "conceptual"
        graph_ctx = comps["logic_retriever"].retrieve_subgraph(topic, hops=hops, question_type=question_type)
        ctx = graph_ctx
        raw_json, method = gen.generate(topic=topic, graph_context=graph_ctx, qb_retriever=qb, question_format=question_format)

    try:
        data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except Exception:
        data = {"raw": raw_json}

    # Attach metadata
    data["topic"] = topic
    data["generator"] = generator
    data["method"] = method
    data["question_format"] = question_format
    data["retrieved_context"] = ctx

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate a single question for a single topic.")
    parser.add_argument("--topic", required=True, help="Topic or query string")
    parser.add_argument("--generator", default="graph_rag", choices=["no_retrieval","vector_rag","graph_rag"], help="Which generation strategy to use")
    parser.add_argument("--format", default="mcq_single", choices=["mcq_single","mcq_multi","true_false","fill_blank","open_answer"], help="Question format")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k for vector retrieval")
    parser.add_argument("--hops", type=int, default=2, help="Graph hops for graph retriever")
    parser.add_argument("--output", help="Optional path to save JSON output")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not set in environment.")
        sys.exit(1)

    res = generate_single(api_key, args.topic, generator=args.generator, question_format=args.format, top_k=args.top_k, hops=args.hops)

    out_str = json.dumps(res, ensure_ascii=False, indent=2)
    # Always print the final generated question JSON to stdout (single line of output)
    print(out_str)
    # Also save to file if requested, but do not print extra messages to stdout
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(out_str)
            # Log the save action to the experiment log
            try:
                comps = _init_shared(api_key)
                comps['logger'].log(f"Saved single_query result to {args.output}")
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] Failed to save output: {e}")


if __name__ == "__main__":
    main()
