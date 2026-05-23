"""kg_loader.py
==============
One-time initialisation of KG, retrievers, and question bank encoder.

两种用法：
  1) 作为库导入： from kg_loader import init_shared
  2) 作为守护进程单独启动： python kg_loader.py [--socket /tmp/kg_loader.sock]

Usage as library:
  from kg_loader import init_shared, TEXTBOOK_DIR, TRIPLETS_PATH, QUESTION_BANK_PATH

  comps = init_shared(api_key)
  comps["kg"]               # AdvancedKnowledgeGraph
  comps["logger"]           # Logger
  comps["vector_retriever"] # VectorBaselineRetriever
  comps["logic_retriever"]  # LogicGraphRetriever
  comps["qb_retriever"]     # QuestionBankRetriever
"""
import os
import sys
import json
import struct
import time
import argparse
import traceback
from dotenv import load_dotenv

load_dotenv()

from rag_system.knowledge_graph import AdvancedKnowledgeGraph
from rag_system.retriever import VectorBaselineRetriever, LogicGraphRetriever, QuestionBankRetriever
from rag_system.generator import NoRetrievalGenerator, BaselineGenerator, SmartGenerator
from rag_system.logger import Logger
from sentence_transformers import SentenceTransformer

# Default paths (same as run_all.py)
TEXTBOOK_DIR      = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH     = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"

DEFAULT_SOCKET_PATH = "/tmp/kg_loader.sock"


def init_shared(api_key: str, log_dir: str = "single_query_logs"):
    """Load KG, retrievers, and question bank encoder into memory.

    Returns a dict with keys:
        logger, kg, vector_retriever, logic_retriever, qb_retriever
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

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
    logic_retriever  = LogicGraphRetriever(kg, vector_retriever)

    # Question bank encoder + retriever (few-shot)
    encoder      = SentenceTransformer("all-MiniLM-L6-v2")
    qb_retriever = QuestionBankRetriever(QUESTION_BANK_PATH, encoder)

    return {
        "logger":           logger,
        "kg":               kg,
        "vector_retriever": vector_retriever,
        "logic_retriever":  logic_retriever,
        "qb_retriever":     qb_retriever,
    }


# ── 以下为 Unix socket daemon 模式 ──

def _generate_single(topic: str, generator: str, question_format: str,
                     top_k: int, hops: int, comps: dict, api_key: str) -> dict:
    """Generate one question (mirrors single_query.generate_single logic)."""
    qb = comps["qb_retriever"]
    timings = {}

    if generator == "no_retrieval":
        t0 = time.time()
        gen = NoRetrievalGenerator(api_key=api_key)
        raw_json, method = gen.generate(topic=topic, qb_retriever=qb,
                                        question_format=question_format)
        timings["gen_time"] = round(time.time() - t0, 2)
        ctx = []
        timings["retrieval_time"] = 0.0
    elif generator == "vector_rag":
        t0 = time.time()
        ctx = comps["vector_retriever"].retrieve(topic, top_k=top_k)
        timings["retrieval_time"] = round(time.time() - t0, 2)
        t0 = time.time()
        gen = BaselineGenerator(api_key=api_key)
        raw_json, method = gen.generate(topic=topic, context=ctx, qb_retriever=qb,
                                        question_format=question_format)
        timings["gen_time"] = round(time.time() - t0, 2)
    else:  # graph_rag
        gen = SmartGenerator(api_key=api_key)
        question_type = ("computational"
                         if any(k in topic.lower()
                                for k in ("sort","hash","tree","bfs","dfs","dynamic",
                                          "knapsack","recursion","dijkstra",
                                          "binary search","heap"))
                         else "conceptual")
        t0 = time.time()
        graph_ctx = comps["logic_retriever"].retrieve_subgraph(
            topic, hops=hops, question_type=question_type)
        timings["retrieval_time"] = round(time.time() - t0, 2)
        ctx = graph_ctx
        t0 = time.time()
        raw_json, method = gen.generate(topic=topic, graph_context=graph_ctx,
                                        qb_retriever=qb, question_format=question_format)
        timings["gen_time"] = round(time.time() - t0, 2)

    try:
        data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except Exception:
        data = {"raw": raw_json}

    data["topic"] = topic
    data["generator"] = generator
    data["method"] = method
    data["question_format"] = question_format
    data["retrieved_context"] = ctx
    data["timings"] = timings
    return data


def _extract_minimal(qjson: dict, requested_format: str = "mcq_single") -> dict:
    """Extract {question, answer} from a generated question JSON."""
    qfmt = requested_format
    out_q = None
    out_a = None

    if qfmt == 'mcq_single':
        out_q = qjson.get('question')
        out_a = qjson.get('correct_answer') or (
            qjson.get('correct_answers') and qjson.get('correct_answers')[0])
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


def _handle_request(request: dict, comps: dict, api_key: str) -> dict:
    """Process one query and return minimal result + timing."""
    topic = request.get('topic')
    if not topic:
        return {"error": "missing 'topic' field"}

    generator = request.get('generator', 'graph_rag')
    question_format = request.get('question_format') or request.get('format') or 'mcq_single'
    top_k = int(request.get('top_k', 5))
    hops = int(request.get('hops', 2))

    t_start = time.time()
    try:
        full = _generate_single(topic, generator, question_format, top_k, hops, comps, api_key)
        total_time = round(time.time() - t_start, 2)
        minimal = _extract_minimal(full, requested_format=question_format)
        minimal["timings"] = {
            "total":      total_time,
            "retrieval":  full.get("timings", {}).get("retrieval_time", 0.0),
            "generation": full.get("timings", {}).get("gen_time", 0.0),
        }
        minimal["method"] = full.get("method", "unknown")
        return minimal
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def _send_msg(sock, msg: bytes):
    """Send length-prefixed message over a socket."""
    sock.sendall(struct.pack(">I", len(msg)) + msg)


def _recv_msg(sock) -> bytes:
    """Receive length-prefixed message from a socket."""
    raw_len = sock.recv(4)
    if not raw_len:
        return b""
    msg_len = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            break
        data += chunk
    return data


def run_server(socket_path: str = DEFAULT_SOCKET_PATH, api_key: str = None):
    """Start Unix socket daemon that keeps KG in memory."""
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # Remove stale socket file
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    print(f"[kg_loader] Loading KG, model, and question bank (one-time cost)…", file=sys.stderr)
    comps = init_shared(api_key)
    print(f"[kg_loader] Ready. Listening on {socket_path}", file=sys.stderr)

    import socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = _recv_msg(conn)
                if not data:
                    continue
                request = json.loads(data.decode())
                result = _handle_request(request, comps, api_key)
                resp = json.dumps(result, ensure_ascii=False).encode()
                try:
                    _send_msg(conn, resp)
                except BrokenPipeError:
                    pass  # client disconnected before we could respond
            except Exception as e:
                try:
                    err = json.dumps({"error": str(e)}).encode()
                    _send_msg(conn, err)
                except Exception:
                    pass  # don't crash on send errors
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        print(f"\n[kg_loader] Shutting down.", file=sys.stderr)
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG loader daemon — keeps KG in memory")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH,
                        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})")
    args = parser.parse_args()
    run_server(socket_path=args.socket)
