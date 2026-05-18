import os
import json
from collections import defaultdict
from typing import Dict

from .logger import Logger


class AdvancedKnowledgeGraph:
    def __init__(self, logger: Logger):
        self.nodes          = {}
        self.edges          = defaultdict(list)
        self.node_metadata  = {}
        self.entity_to_nodes = defaultdict(set)
        self.logger         = logger

        self.stop_entities = {
            "algorithm", "data", "computer", "problem", "value", "system",
            "number", "example", "result", "process", "method", "function",
            "type", "set", "analysis", "information", "code", "input", "output",
            "concept", "step", "theory", "approach", "detail"
        }

    # ------------------------------------------------------------------
    # Basic graph operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, content: str, metadata: dict = None):
        self.nodes[node_id] = content
        if metadata:
            self.node_metadata[node_id] = metadata

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str = "related",
        head_label: str = None,   # ← FIX: entity name for from_id side
        tail_label: str = None,   # ← FIX: entity name for to_id side
    ):
        """Add an undirected edge (both directions stored).

        head_label / tail_label are the human-readable entity strings from
        the LLM-extracted triplet.  When present they are surfaced in
        retrieve_subgraph so the generator prompt contains real semantic
        information instead of node IDs.
        """
        if from_id in self.nodes and to_id in self.nodes:
            self.edges[from_id].append({
                'to':         to_id,
                'relation':   relation,
                'head_label': head_label or from_id,
                'tail_label': tail_label or to_id,
            })
            # reversed direction
            self.edges[to_id].append({
                'to':         from_id,
                'relation':   relation,
                'head_label': tail_label or to_id,
                'tail_label': head_label or from_id,
            })

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def build_base_structure(self, textbook_dir: str):
        total_nodes = 0
        for i in range(1, 21):
            tb_path = os.path.join(
                textbook_dir, f"textbook{i}", f"textbook{i}_structured.json"
            )
            if not os.path.exists(tb_path):
                continue
            with open(tb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for j, item in enumerate(data):
                node_id = f"tb{i}_node{j}"
                self.add_node(node_id, str(item.get('content', '')), item)
                if j > 0:
                    # structural edges have no meaningful entity labels
                    self.add_edge(f"tb{i}_node{j-1}", node_id, "structural")
                total_nodes += 1
        self.logger.log(f"   >>> Loaded {total_nodes} base knowledge slices.")

    def load_triplets(self, triplets_path: str):
        """Load LLM-extracted triplets and create logic edges.

        FIX: entity labels (head / tail text) are now stored on every
        logic edge, so retrieve_subgraph can expose them to the generator.
        """
        if not os.path.exists(triplets_path):
            self.logger.log(f"   [Error] Knowledge graph file not found: {triplets_path}")
            return

        with open(triplets_path, 'r', encoding='utf-8') as f:
            data     = json.load(f)
            triplets = data.get('triplets', [])
        self.logger.log(f"   >>> Loading {len(triplets)} logic edges...")

        # If textbooks weren't loaded, there may be no node content for
        # the source_node ids referenced by triplets. Collect head/tail
        # strings per source_node and create lightweight node content so
        # graph expansion and retrieval still work.
        node_texts = {}
        for t in triplets:
            src = t.get('source_node')
            if not src:
                continue
            head = t.get('head', '')
            tail = t.get('tail', '')
            # accumulate short representative text
            node_texts.setdefault(src, []).append(f"{head}. {tail}.")

        # Add node placeholders for any referenced source_node missing
        for nid, snippets in node_texts.items():
            if nid not in self.nodes:
                content = " ".join(snippets)[:1000]
                self.add_node(nid, content, metadata={"inferred_from_triplets": True})

        valid_triplets = []
        for t in triplets:
            if not all(k in t for k in ('head', 'tail', 'relation', 'source_node')):
                continue
            h    = t['head'].lower()
            tail = t['tail'].lower()
            src  = t['source_node']
            if h in self.stop_entities or tail in self.stop_entities:
                continue
            self.entity_to_nodes[h].add(src)
            self.entity_to_nodes[tail].add(src)
            valid_triplets.append(t)

        edge_count = 0
        for t in valid_triplets:
            h    = t['head'].lower()
            tail = t['tail'].lower()
            src  = t['source_node']
            related = self.entity_to_nodes[h] | self.entity_to_nodes[tail]
            for target in related:
                if target != src and target in self.nodes:
                    # ← FIX: pass the original entity strings as labels
                    self.add_edge(
                        src, target,
                        relation=t['relation'],          # drop "logic_" prefix – cleaner
                        head_label=t['head'],
                        tail_label=t['tail'],
                    )
                    edge_count += 1
            if edge_count > 300_000:
                break

        self.logger.log(f"   >>> Built {edge_count} logic association paths.")

    def load_question_bank(self, question_bank_path: str):
        """
        Load question bank and add each question as a lightweight KG node.
        This lets retrievers fall back to QBank contents when textbooks
        are unavailable.
        """
        if not os.path.exists(question_bank_path):
            self.logger.log(f"   [!] Question bank not found at {question_bank_path} — skipping KG import.")
            return

        try:
            with open(question_bank_path, 'r', encoding='utf-8') as fh:
                questions = json.load(fh)
        except Exception:
            self.logger.log(f"   [!] Failed to parse question bank: {question_bank_path}")
            return

        added = 0
        for q in questions:
            qid = q.get('id') or q.get('qid') or q.get('unique_id')
            if not qid:
                continue
            nid = f"qbank_{qid}"
            if nid in self.nodes:
                continue
            # Use question text + rationale as node content
            text = q.get('question') or q.get('statement') or q.get('sentence') or ''
            rationale = q.get('rationale') or q.get('explanation') or ''
            content = (text + ' ' + rationale).strip()[:1000]
            self.add_node(nid, content, metadata={"source": "question_bank"})
            added += 1

        self.logger.log(f"   >>> Imported {added} question-bank items into KG as nodes.")
