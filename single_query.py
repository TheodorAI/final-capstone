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
    parser = argparse.ArgumentParser(description="Generate a single question from JSON input and output minimal JSON result.")
    parser.add_argument("--input", help="Path to input JSON file (use '-' or omit to read from stdin).")
    parser.add_argument("--output", help="Optional path to save FULL JSON output (silent).")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not set in environment.")
        sys.exit(1)

    # Read input JSON from file or stdin
    input_json = None
    if args.input and args.input != '-':
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                input_json = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read input JSON: {e}")
            sys.exit(1)
    else:
        try:
            raw = sys.stdin.read()
            input_json = json.loads(raw) if raw and raw.strip() else {}
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON from stdin: {e}")
            sys.exit(1)

    topic = input_json.get('topic')
    if not topic:
        print('[ERROR] input JSON must contain "topic" field')
        sys.exit(1)

    generator = input_json.get('generator', 'graph_rag')
    question_format = input_json.get('question_format') or input_json.get('format') or 'mcq_single'
    top_k = int(input_json.get('top_k', 5))
    hops = int(input_json.get('hops', 2))

    # Initialise shared components and generate
    comps = _init_shared(api_key)
    res = generate_single(api_key, topic, generator=generator, question_format=question_format, top_k=top_k, hops=hops, comps=comps)

    def _extract_minimal(qjson: dict) -> dict:
        # Validate: warn if actual format differs from requested
        actual_fmt = qjson.get('question_format')
        if actual_fmt and actual_fmt != question_format:
            try:
                comps['logger'].log(f"[WARN] Format mismatch: requested='{question_format}' but got='{actual_fmt}' for topic='{topic}'")
            except Exception:
                pass
        
        # Force requested format for extraction (ignore what question claims)
        qfmt = question_format
        out_q = None
        out_a = None

        if qfmt == 'mcq_single':
            out_q = qjson.get('question')
            out_a = qjson.get('correct_answer') or (qjson.get('correct_answers') and qjson.get('correct_answers')[0])
        elif qfmt == 'mcq_multi':
            out_q = qjson.get('question')
            ca = qjson.get('correct_answers') or qjson.get('correct_answer')
            out_a = ','.join(ca) if isinstance(ca, list) else str(ca)
        elif qfmt == 'true_false':
            out_q = qjson.get('statement')
            out_a = str(qjson.get('tf_answer'))
        elif qfmt == 'fill_blank':
            out_q = qjson.get('sentence')
            ans = qjson.get('answers') or []
            out_a = ' | '.join(ans) if isinstance(ans, list) else str(ans)
        elif qfmt == 'open_answer':
            out_q = qjson.get('question')
            out_a = qjson.get('model_answer') or qjson.get('answer')
        else:
            out_q = qjson.get('question') or qjson.get('sentence') or qjson.get('statement')
            out_a = qjson.get('answer') or qjson.get('model_answer') or qjson.get('correct_answer')

        if not out_q and isinstance(qjson.get('question'), dict):
            nested = qjson.get('question')
            out_q = nested.get('question') or nested.get('statement') or nested.get('sentence')
            out_a = nested.get('answer') or nested.get('correct_answer') or nested.get('model_answer')

        return {"question": out_q or "", "answer": out_a or ""}

    minimal = _extract_minimal(res)

    # Output minimal JSON (only question and answer)
    print(json.dumps(minimal, ensure_ascii=False))

    # Optionally save full JSON silently and log
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            try:
                comps['logger'].log(f"Saved single_query result to {args.output}")
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] Failed to save output: {e}")


if __name__ == "__main__":
    main()
