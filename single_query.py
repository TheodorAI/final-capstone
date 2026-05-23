"""single_query.py
=================
轻量客户端 — 连接到 kg_loader daemon 生成题目，无需重复加载 KG。

Usage:
  # 先启动 daemon（只需一次）：
  python kg_loader.py

  # 然后查询：
  python single_query.py --topic "merge sort"
  python single_query.py --topic "BFS" --generator vector_rag --format mcq_single

  # 从 stdin 传参：
  echo '{"topic":"merge sort","generator":"graph_rag"}' | python single_query.py
"""
import os
import sys
import json
import struct
import argparse
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SOCKET_PATH = "/tmp/kg_loader.sock"


def _send_msg(sock, msg: bytes):
    sock.sendall(struct.pack(">I", len(msg)) + msg)


def _recv_msg(sock) -> bytes:
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


def query_daemon(request: dict, socket_path: str = DEFAULT_SOCKET_PATH) -> dict:
    """Send a query to the kg_loader daemon and return the result."""
    import socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        _send_msg(sock, json.dumps(request).encode())
        raw = _recv_msg(sock)
        return json.loads(raw.decode()) if raw else {"error": "empty response"}
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single question via kg_loader daemon."
    )
    parser.add_argument("--topic", help="Topic to generate a question for.")
    parser.add_argument("--generator", choices=["no_retrieval", "vector_rag", "graph_rag"],
                        default="graph_rag")
    parser.add_argument("--format", dest="question_format",
                        choices=["mcq_single", "mcq_multi", "true_false",
                                 "fill_blank", "open_answer"],
                        default="mcq_single")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--input", help="Path to input JSON file (alternative to --topic).")
    parser.add_argument("--output", help="Optional path to save full result JSON.")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH,
                        help=f"kg_loader socket path (default: {DEFAULT_SOCKET_PATH})")
    args = parser.parse_args()

    # Build request dict
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            request = json.load(f)
    elif args.topic:
        request = {
            "topic": args.topic,
            "generator": args.generator,
            "question_format": args.question_format,
            "top_k": args.top_k,
            "hops": args.hops,
        }
    else:
        # Try reading from stdin
        try:
            raw = sys.stdin.read()
            request = json.loads(raw) if raw.strip() else {}
        except Exception as e:
            print(f"[ERROR] No input provided. Use --topic or --input or pipe JSON to stdin.")
            sys.exit(1)

    if not request.get("topic"):
        print('[ERROR] "topic" is required.')
        sys.exit(1)

    # Check daemon is running
    if not os.path.exists(args.socket):
        print(f"[ERROR] kg_loader not running. Start it first:\n"
              f"  python kg_loader.py --socket {args.socket}",
              file=sys.stderr)
        sys.exit(1)

    result = query_daemon(request, socket_path=args.socket)

    if "error" in result:
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    # Output question and answer
    print(f"题目: {result.get('question', '')}")
    print(f"答案: {result.get('answer', '')}")

    # Output timing breakdown
    timings = result.get("timings", {})
    total = timings.get("total", 0)
    retrieval = timings.get("retrieval", 0)
    gen = timings.get("generation", 0)
    method = result.get("method", "unknown")
    print(f"\n生成方式: {method}")
    print(f"总耗时:   {total:.1f}s")
    print(f"  ├─ 检索: {retrieval:.1f}s")
    print(f"  └─ 生成: {gen:.1f}s (含 LLM 调用 + 难度过滤)")

    # Also output compact JSON line
    print(f"\n(JSON) {json.dumps(result, ensure_ascii=False)}")

    # Optionally save full result
    if args.output:
        try:
            # Re-query to get full result
            request["_full"] = True
            full = query_daemon(request, socket_path=args.socket)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(full, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save output: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
