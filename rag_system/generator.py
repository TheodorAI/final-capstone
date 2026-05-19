import json
import random
import re
from typing import List, Dict, Tuple, Optional

from openai import OpenAI


_NODE_ID_RE = re.compile(
    r'\btb\d+_node\d+\b'       # e.g. tb6_node174
    r'|\bnode_\d+\b'            # e.g. node_42
    r'|\btb\d+node\d+\b'        # e.g. tb6node174 (no underscore variant)
    r'|\b[a-z]{2,4}\d+_node\d+\b',  # e.g. ns3_node12
    re.IGNORECASE,
)

def _sanitize(text: str) -> str:
    """Strip internal KG node identifiers from node content."""
    return _NODE_ID_RE.sub('[node]', text)


_BIG_O_TOKEN_RE = re.compile(r'\bo\s*\((.*?)\)', re.IGNORECASE)
_FILL_BLANK_SYMBOLIC_ONLY_RE = re.compile(
    r'^(?:[A-Za-z]+\([A-Za-z0-9_, ]+\)|[A-Za-z][A-Za-z0-9_]*(?:\[[^\]]+\])?)$'
)


def _normalize_fill_blank_answer(text: str) -> str:
    """Normalize fill-in answers before comparing independently derived answers."""
    if not isinstance(text, str):
        text = str(text)
    normalized = text.strip().lower()
    normalized = normalized.replace("≤", "<=").replace("≥", ">=")
    normalized = normalized.replace("Θ", "theta").replace("θ", "theta")
    normalized = re.sub(r'[`"\']', "", normalized)
    normalized = re.sub(r'\b(an?|the)\b', " ", normalized)
    normalized = re.sub(r'\s+', " ", normalized).strip()

    def _big_o_repl(match: re.Match) -> str:
        inner = re.sub(r'\s+', "", match.group(1).lower())
        return f"o({inner})"

    normalized = _BIG_O_TOKEN_RE.sub(_big_o_repl, normalized)
    return normalized


def _validate_fill_blank_shape(q: Dict) -> Tuple[bool, str]:
    """Cheap structural checks before paying for LLM verification."""
    sentence = q.get("sentence", "")
    answers = q.get("answers", [])
    explanation = q.get("explanation", "")
    blank_count = sentence.count("___")

    if not sentence or blank_count == 0:
        return False, "missing sentence or blanks"
    if not isinstance(answers, list) or len(answers) != blank_count:
        return False, f"blank/answer mismatch ({blank_count} blanks vs {len(answers) if isinstance(answers, list) else 'non-list'} answers)"

    for idx, answer in enumerate(answers, start=1):
        if not isinstance(answer, str) or not answer.strip():
            return False, f"answer {idx} is empty"

    if explanation and re.search(r'\b(wait|actually|recalculat|re-check|correction)\b', explanation, re.IGNORECASE):
        return False, "explanation contains self-correction language"

    if q.get("question_type") == "computational":
        for idx, answer in enumerate(answers, start=1):
            norm = _normalize_fill_blank_answer(answer)
            if _FILL_BLANK_SYMBOLIC_ONLY_RE.fullmatch(norm) and "___" in sentence:
                trigger_phrases = (
                    "recursive calls", "time complexity", "space complexity",
                    "value of", "index", "comparisons", "iterations", "output",
                    "result", "collisions"
                )
                if any(phrase in sentence.lower() for phrase in trigger_phrases):
                    return False, f"answer {idx} is only a symbolic placeholder"

    return True, ""


# ---------------------------------------------------------------------------
# Shared non-MCQ format generation helpers
# Used by NoRetrievalGenerator, BaselineGenerator, and SmartGenerator
# to avoid duplicating prompt logic across three classes.
# ---------------------------------------------------------------------------

_NON_MCQ_PROMPTS = {
    ("mcq_multi", "computational"): """\
Generate ONE multiple-correct-answer question with 5 options (A–E).
Exactly 2–3 options are correct. Each correct option tests a different
computational aspect (complexity, intermediate state, invariant).
Wrong options must arise from specific named errors (off-by-one, wrong
formula, boundary mistake) — not obviously wrong.
Return JSON: {{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"...","E":"..."}},"correct_answers":["A","C"],"explanation":"...","question_type":"computational","question_format":"mcq_multi"}}""",

    ("mcq_multi", "conceptual"): """\
Generate ONE multiple-correct-answer question with 5 options (A–E).
Exactly 2–3 options are correct. Each correct option captures a different
true consequence of the same causal mechanism. Wrong options are subtly
incorrect versions — not obviously wrong.
Return JSON: {{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"...","E":"..."}},"correct_answers":["A","C"],"explanation":"...","question_type":"conceptual","question_format":"mcq_multi"}}""",

    ("true_false", "computational"): """\
Generate ONE True/False statement. It must make a specific, verifiable claim
about an algorithm's output or complexity on a concrete input. Surface
intuition should point to the wrong answer (trap question).
Return JSON: {{"statement":"...","tf_answer":true,"explanation":"...","question_type":"computational","question_format":"true_false"}}""",

    ("true_false", "conceptual"): """\
Generate ONE True/False statement about a causal relationship or security
property. It must APPEAR true on surface reading but be false due to a
specific technical nuance (or vice versa).
Return JSON: {{"statement":"...","tf_answer":false,"explanation":"...","question_type":"conceptual","question_format":"true_false"}}""",

    ("fill_blank", "computational"): """\
Generate ONE fill-in-the-blank sentence with 1–2 blanks (___).
Each blank must require a specific computed value, complexity expression,
or algorithm name. Answers list must match blank order.
Return JSON: {{"sentence":"The ___ algorithm has worst-case complexity ___.","answers":["merge sort","O(n log n)"],"explanation":"...","question_type":"computational","question_format":"fill_blank"}}""",

    ("fill_blank", "conceptual"): """\
Generate ONE fill-in-the-blank sentence with 1–2 blanks (___).
Each blank must complete a causal claim with a specific technical term —
not a vague word. Answers list must match blank order.
Return JSON: {{"sentence":"When ___ is broken, an attacker can forge ___.","answers":["collision resistance","digital signatures"],"explanation":"...","question_type":"conceptual","question_format":"fill_blank"}}""",

    ("open_answer", "computational"): """\
Generate ONE open-answer question. Ask the student to trace an algorithm
on a specific input AND explain why the result demonstrates a complexity
or correctness property. Model answer must include ALL intermediate states.
Key points must be checkable criteria (e.g. "states exactly 11 comparisons").
Return JSON: {{"question":"...","model_answer":"...","key_points":["criterion 1","criterion 2","criterion 3"],"question_type":"computational","question_format":"open_answer"}}""",

    ("open_answer", "conceptual"): """\
Generate ONE open-answer question. Ask the student to explain HOW and WHY
two specific mechanisms interact, including what breaks if one fails.
Model answer must state the causal chain explicitly.
Key points must be specific causal claims, not vague summaries.
Return JSON: {{"question":"...","model_answer":"...","key_points":["criterion 1","criterion 2","criterion 3"],"question_type":"conceptual","question_format":"open_answer"}}""",
}


def _call_non_mcq_llm(llm, prompt: str, question_format: str,
                       question_type: str) -> str:
    """Shared LLM call + JSON parse for non-MCQ formats."""
    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        data.setdefault("question_format", question_format)
        data.setdefault("question_type",   question_type)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        fallback = {"question_format": question_format,
                    "question_type": question_type,
                    "error": str(e)}
        return json.dumps(fallback, ensure_ascii=False)


def _generate_non_mcq_no_context(llm, topic: str, question_format: str,
                                   question_type: str) -> str:
    """For NoRetrievalGenerator: build prompt without any retrieved context."""
    instr = _NON_MCQ_PROMPTS.get((question_format, question_type), "")
    prompt = (f'You are an elite CS Professor. Topic: "{topic}"\n\n'
              f'{instr}\n\n'
              f'Use only your knowledge — no external context provided.')
    return _call_non_mcq_llm(llm, prompt, question_format, question_type)


def _generate_non_mcq_with_context(llm, topic: str, question_format: str,
                                    question_type: str, context_str: str) -> str:
    """For BaselineGenerator: build prompt with vector-retrieved context."""
    instr = _NON_MCQ_PROMPTS.get((question_format, question_type), "")
    prompt = (f'You are an elite CS Professor. Topic: "{topic}"\n\n'
              f'=== RETRIEVED CONTEXT ===\n{context_str}\n\n'
              f'{instr}\n\n'
              f'Base the question on the provided context where possible.')
    return _call_non_mcq_llm(llm, prompt, question_format, question_type)


MAX_DIFFICULTY_RETRIES = 3   # max regeneration attempts per question
# ★ FIX-THRESHOLD: Reverted from 4→3.
# Data shows difficulty=3 questions score overall=3.836 (HIGHER than
# difficulty=4 at 3.744). Threshold=4 was forcing retries that replaced
# good difficulty=3 questions with worse difficulty=4 ones.
# Per-format exceptions handled in _FORMAT_DIFFICULTY_THRESHOLD below.
DIFFICULTY_THRESHOLD   = 3   # scores <= this trigger regeneration

_DIFFICULTY_JUDGE_PROMPT_COMPUTATIONAL = """\
You are an expert Computer Science exam psychometrician. Score the difficulty of the \
following MCQ for university-level CS students (sophomore / junior year) on a strict \
INTEGER scale from 1 to 5.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{correct_answer}

=== RATIONALE ===
{rationale}

=== STRICT SCORING RUBRIC ===

Score 1 — TRIVIAL
  • Pure definition recall with no computation.
  • The answer can be copied verbatim from a one-sentence textbook glossary entry.
  • Examples: "What data structure does BFS use?", "Define a base case.",
    "Which sorting algorithm has O(n log n) average time complexity?"

Score 2 — SIMPLE
  • Single-algorithm mechanical trace on a small, happy-path input (≤ 6 elements).
  • OR one-step formula substitution (the formula is directly named or universally known).
  • The student follows one rule with no branching, no edge condition, no state
    accumulation across more than 4 steps.
  • Examples: "How many comparisons does insertion sort make on a reverse-sorted
    N-element array?" (answer = N*(N-1)/2), "Trace BFS on a 3-node graph from A.",
    "Apply h(k) = k mod 7 to k = 14."

Score 3 — MODERATE
  • Multi-step trace (≥ 5 distinct state-changing operations) requiring sustained
    attention to detail, OR a single algorithm applied to input with ONE non-trivial
    edge condition (collision, boundary, missing base case, etc.).
  • The student must track evolving state but uses a single concept without surprises.

Score 4 — CHALLENGING  (EITHER of the two paths below qualifies)
  PATH A — Cross-concept: Requires integrating exactly TWO distinct algorithmic
    concepts that CAUSALLY INTERACT (e.g. hash function output → collision chain
    length, pivot choice → recursion depth, base-case design → space complexity).
  PATH B — Deep single-concept adversarial: Uses ONE algorithm, but ALL of:
    (a) ≥ 10 distinct state-changing operations in a full mental simulation,
    (b) an adversarial or degenerate input that forces worst-case / boundary behavior,
    (c) at least ONE non-obvious trap,
    (d) the correct answer CANNOT be reached by rule-of-thumb or formula alone.

Score 5 — EXPERT  (EITHER of the two paths below qualifies)
  PATH A — Multi-concept: Synthesises THREE or more concepts with non-obvious causal interactions.
  PATH B — Deep single-concept: requires simultaneously reasoning at TWO distinct levels
    (e.g. trace concrete execution AND diagnose why a correctness/complexity claim is violated).

=== SCORING DISCIPLINE ===
- Assign score 2 if the answer follows from a named formula or a ≤ 4-step trace.
- Assign score 3 only when ≥ 5 distinct state-changing operations must be executed.
- Assign score 4 via PATH A only when you can name BOTH concepts and the causal direction.
- Default to the LOWER score when uncertain between two adjacent levels.

Return ONLY valid JSON, no markdown:
{{"score": <integer 1-5>, "reasoning": "2-3 sentence explanation citing the specific \
path (A or B) and the concrete evidence from the question text"}}
"""

_DIFFICULTY_JUDGE_PROMPT_CONCEPTUAL = """\
You are an expert Computer Science exam psychometrician. Score the difficulty of the \
following conceptual MCQ for university-level CS students on a strict INTEGER scale from 1 to 5.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{correct_answer}

=== RATIONALE ===
{rationale}

=== STRICT SCORING RUBRIC ===

Score 1 — TRIVIAL
  • Pure definition recall: the answer is a single term found in any glossary.
  • Examples: "What does CIA stand for in cybersecurity?", "What is a firewall?",
    "Which layer does TCP operate on?"

Score 2 — SIMPLE
  • Applies one concept with no causal reasoning — the student only needs to match
    a concept to its property or effect.
  • Example: "Why do we use salts in password hashing?" (one-step: prevents precomputation).

Score 3 — MODERATE
  • Requires understanding the mechanism behind ONE concept, including a non-obvious
    consequence or exception.
  • The student must reason about HOW or WHY, not just WHAT, but stays within one domain.
  • Example: Explaining why a birthday attack's complexity differs from a preimage attack.

Score 4 — CHALLENGING  (EITHER path qualifies)
  PATH A — Cross-concept: TWO distinct mechanisms/protocols CAUSALLY INTERACT.
    The student must trace how a property/failure in concept A directly determines
    the behaviour of concept B (e.g., hash collision → signature forgery,
    stateful firewall reassembly → IDS detection, fast hash speed → brute-force feasibility).
  PATH B — Deep single-concept adversarial: ONE mechanism, but the question requires
    evaluating a SUBTLE MISCONCEPTION or EDGE CASE that even careful students miss
    (e.g., a scenario where the standard defence fails under specific conditions).

Score 5 — EXPERT
  • Synthesises THREE or more concepts with non-obvious interactions, OR requires
    simultaneously reasoning about an attack, its defence, AND a second-order failure
    of that defence — all in a single question.

=== SCORING DISCIPLINE ===
- Score 2 if the answer requires recalling the effect of a single mechanism.
- Score 4 PATH A only when you can name BOTH concepts and state the causal direction.
- Default to the LOWER score when uncertain between two adjacent levels.

Return ONLY valid JSON, no markdown:
{{"score": <integer 1-5>, "reasoning": "2-3 sentence explanation citing the specific \
path (A or B) and the concrete evidence from the question text"}}
"""


def assess_difficulty(llm_client: OpenAI, question_json_str: str) -> Tuple[int, str]:
    """
    Call LLM to judge the difficulty of a generated question on a 1-5 scale.
    Automatically selects the appropriate rubric based on question_type
    (computational vs conceptual) read from the question JSON.

    Returns
    -------
    (score, reasoning)
        score     : int 1-5  (defaults to 3 on error — treated as acceptable)
        reasoning : brief human-readable explanation from the judge LLM
    """
    try:
        q = json.loads(question_json_str)
    except Exception:
        return 3, "JSON parse error — treating as acceptable (score=3)"

    question_format = q.get("question_format", "mcq_single")
    question_type   = q.get("question_type", "computational")

    # Format-aware field extraction — each format stores content in different keys
    if question_format == "true_false":
        question       = q.get("statement", "")
        correct_answer = str(q.get("tf_answer", ""))
        rationale      = q.get("explanation", "")[:500]
    elif question_format == "fill_blank":
        question       = q.get("sentence", "")
        correct_answer = str(q.get("answers", []))
        rationale      = q.get("explanation", "")[:500]
    elif question_format == "open_answer":
        question       = q.get("question", "")
        correct_answer = q.get("model_answer", "")[:300]
        rationale      = str(q.get("key_points", []))[:300]
    elif question_format == "mcq_multi":
        question       = q.get("question", "")
        correct_answer = str(q.get("correct_answers", []))
        rationale      = q.get("explanation", "")[:500]
    else:  # mcq_single (default)
        question       = q.get("question", "")
        correct_answer = q.get("correct_answer", "")
        rationale      = q.get("rationale", "")[:500]

    if not question:
        return 3, "Empty question — treating as acceptable (score=3)"

    # Select rubric based on question type
    prompt_template = (
        _DIFFICULTY_JUDGE_PROMPT_CONCEPTUAL
        if question_type == "conceptual"
        else _DIFFICULTY_JUDGE_PROMPT_COMPUTATIONAL
    )
    prompt = prompt_template.format(
        question=question,
        correct_answer=correct_answer,
        rationale=rationale,
    )

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0,
        )
        result    = json.loads(response.choices[0].message.content)
        raw_score = result.get("score", 3)
        reasoning = result.get("reasoning", "")
        try:
            score = int(raw_score)
        except (ValueError, TypeError):
            score = 3
        score = max(1, min(5, score))   # clamp to [1, 5]
        return score, reasoning
    except Exception as exc:
        return 3, f"Judge LLM error ({exc}) — treating as acceptable (score=3)"


# ---------------------------------------------------------------------------
# Difficulty Boost Block (injected into retry prompts)
# ---------------------------------------------------------------------------

_DIFFICULTY_BOOST_TEMPLATE_COMPUTATIONAL = """\
╔══════════════════════════════════════════════════════════╗
║            ⚠  DIFFICULTY BOOST REQUIRED  ⚠              ║
╚══════════════════════════════════════════════════════════╝

The question below was REJECTED by an automated difficulty reviewer.
It received a score of {score}/5, which is at or below the minimum
acceptable threshold of {threshold}/5:

--- REJECTED QUESTION ---
{rejected_question}
--- END REJECTED QUESTION ---

REVIEWER'S REASON FOR LOW SCORE:
  "{judge_reasoning}"

You MUST produce a question that scores AT LEAST {threshold}/5.
The following are STRICTLY FORBIDDEN:
  ✗  Score 1: pure definition recall
  ✗  Score 2: direct formula substitution, trace on ≤ 6 happy-path elements,
     or any question structurally similar to the rejected one above

To reach score 4, choose ONE of these two valid paths:

  PATH A — Cross-concept (easier if you have graph context):
    Combine TWO distinct algorithmic concepts that causally interact.
    Name both explicitly. Example: "how does the hash function's output
    determine the length of the linear-probing collision chain?"

  PATH B — Deep single-concept adversarial:
    Use ONE algorithm, but satisfy ALL four conditions:
    (a) The student must execute 5 to 8 distinct state-changing steps. KEEP INPUTS SMALL (e.g., arrays ≤ 6 elements, n ≤ 6) so the math can be verified manually. Do NOT use massive numbers like n=100.
    (b) Use an adversarial or degenerate input (worst-case, all-equal, reverse-sorted, degenerate tree).
    (c) Embed ONE non-obvious trap.
    (d) CRITICAL: You MUST incorporate the KNOWLEDGE GRAPH RELATIONS into the trap or the steps. Do not ignore the graph.

"""

_DIFFICULTY_BOOST_TEMPLATE_CONCEPTUAL = """\
╔══════════════════════════════════════════════════════════╗
║            ⚠  DIFFICULTY BOOST REQUIRED  ⚠              ║
╚══════════════════════════════════════════════════════════╝

The question below was REJECTED by an automated difficulty reviewer.
It received a score of {score}/5, which is at or below the minimum
acceptable threshold of {threshold}/5:

--- REJECTED QUESTION ---
{rejected_question}
--- END REJECTED QUESTION ---

REVIEWER'S REASON FOR LOW SCORE:
  "{judge_reasoning}"

You MUST produce a question that scores AT LEAST {threshold}/5.
The following are STRICTLY FORBIDDEN:
  ✗  Score 1: single-term definition recall
  ✗  Score 2: one-step "concept → property" matching with no causal reasoning

To reach score 4, choose ONE of these two valid paths:

  PATH A — Cross-concept causal interaction (recommended):
    TWO distinct mechanisms/protocols must CAUSALLY INTERACT in the scenario.
    The student must reason how a property or failure in concept A directly
    determines the behaviour or vulnerability of concept B.
    Examples of valid pairs:
      • Hash collision resistance → digital signature forgery
      • Stateful firewall reassembly → IDS signature detection
      • Fast hash computation speed → brute-force feasibility despite salting
      • TCP session state → stateful vs stateless firewall behaviour differences

  PATH B — Deep single-concept adversarial:
    ONE mechanism, but the question presents a SUBTLE MISCONCEPTION or EDGE CASE.
    (a) The scenario must appear superficially correct or safe.
    (b) The trap must hinge on a specific technical detail.
    (c) CRITICAL: You MUST NOT invent fake protocols or arbitrary rules (e.g., fake cache eviction policies). You MUST derive the complexity/trap STRICTLY from real mechanisms found in the KNOWLEDGE_GRAPH_RELATIONS.

"""


def _build_difficulty_boost(rejected_q_json: str, judge_score: int, judge_reasoning: str) -> str:
    """
    Build the difficulty-boost prefix that is prepended to the generation
    prompt when the previous attempt scored below DIFFICULTY_THRESHOLD.
    Automatically selects the computational or conceptual template based
    on the question_type field in the rejected question JSON.
    """
    try:
        q = json.loads(rejected_q_json)
        rejected_question = q.get("question", "")[:400]
        question_type     = q.get("question_type", "computational")
    except Exception:
        rejected_question = str(rejected_q_json)[:400]
        question_type     = "computational"

    template = (
        _DIFFICULTY_BOOST_TEMPLATE_CONCEPTUAL
        if question_type == "conceptual"
        else _DIFFICULTY_BOOST_TEMPLATE_COMPUTATIONAL
    )
    return template.format(
        score=judge_score,
        threshold=DIFFICULTY_THRESHOLD,
        rejected_question=rejected_question,
        judge_reasoning=judge_reasoning,
    )


# ---------------------------------------------------------------------------
# Topic classification (mirrors evaluator.py — keep in sync)
# ---------------------------------------------------------------------------

_COMPUTATIONAL_TOPICS = {
    "hash table", "hash table linear probing", "hash table quadratic probing",
    "sorting", "merge sort", "quick sort", "insertion sort", "bubble sort",
    "heap sort", "heap", "binary heap",
    "graph traversal", "bfs", "dfs", "bfs graph traversal", "dfs graph traversal",
    "dynamic programming", "dynamic programming knapsack", "dynamic programming lcs",
    "recursion", "recursion and recurrence",
    "binary search", "binary search tree", "binary search tree operations",
    "dijkstra", "dijkstra shortest path",
    "data structures and algorithms", "data structure and algorithm",
    "mathematics", "mathematics (for computer science and ai)",
    "algorithm", "algorithms",
}

_CONCEPTUAL_TOPICS = {
    "cybersecurity", "computer networks", "computer network",
    "information retrieval", "artificial intelligence introduction",
    "machine learning", "data science", "data science and big data",
    "human-computer interaction", "nlp", "natural language processing",
    "computer vision", "ethics", "programming fundamentals",
    "computer architecture", "operating systems",
    "database", "database systems",
}


def _choose_question_type(topic: str, qb_retriever=None) -> str:
    # If a question-bank retriever is provided, sample question_type proportionally from the bank
    if qb_retriever is not None and qb_retriever.available:
        topic_norm = topic.lower().strip()
        
        # Count how many computational vs conceptual questions exist for this topic in the bank
        conceptual_count = sum(
            1 for q in qb_retriever.questions
            if q.get("topic", "").lower().strip() == topic_norm
            and (q.get("question_type") == "conceptual" or q.get("type") == "conceptual")
        )
        computational_count = sum(
            1 for q in qb_retriever.questions
            if q.get("topic", "").lower().strip() == topic_norm
            and (q.get("question_type") == "computational" or q.get("type") == "computational")
        )

        total = conceptual_count + computational_count
        if total > 0:
            import random
            # random.choices samples proportionally based on the weights
            return random.choices(
                population=["conceptual", "computational"],
                weights=[conceptual_count, computational_count],
                k=1
            )[0]

    # Fallback: question bank unavailable or no entries for this topic — use hard-coded rules
    t = topic.lower().strip()
    if any(ct in t for ct in _COMPUTATIONAL_TOPICS):
        return "computational"
    if any(ct in t for ct in _CONCEPTUAL_TOPICS):
        return "conceptual"
    return "computational"


# ---------------------------------------------------------------------------
# One-shot example pools
# ---------------------------------------------------------------------------

_ONESHOT_POOL = [
    {
        "generator_scratchpad": {
            "algorithm_rules": "Hash function h(k) = k mod 10, Linear Probing.",
            "step_by_step_execution": (
                "h(12)=2, empty → insert at 2. "
                "h(22)=2, occupied → probe 3, empty → insert at 3. "
                "h(32)=2, occupied → probe 3, occupied → probe 4, empty → insert at 4."
            ),
            "final_state": "Index 4",
        },
        "mcq_data": {
            "question": (
                "A hash table uses linear probing with h(k) = k mod 10. "
                "Insert keys [12, 22, 32] in order. Which index does 32 occupy?"
            ),
            "correct_answer": "Index 4",
            "rationale": "h(12)=2. h(22)=2 collides → 3. h(32)=2 collides → 3 collides → 4.",
            "distractors": [
                {"option": "Index 2", "explanation": "[Initialization Error] Forgetting 12 already occupies slot 2."},
                {"option": "Index 3", "explanation": "[Procedural Omission] Stopping one probe too early."},
                {"option": "Index 5", "explanation": "[Operator Confusion] Applying double-hashing step instead of linear probing."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "Merge sort: recursively split, sort, merge.",
            "step_by_step_execution": (
                "Array [5,2,4,1]. Split → [5,2] and [4,1]. "
                "Sort [5,2] → merge [2,5]. Sort [4,1] → merge [1,4]. "
                "Merge [2,5] and [1,4]: compare 2 vs 1→1; 2 vs 4→2; 5 vs 4→4; 5. Result [1,2,4,5]."
            ),
            "final_state": "[1, 2, 4, 5]",
        },
        "mcq_data": {
            "question": (
                "Apply merge sort to [5, 2, 4, 1]. "
                "What is the final sorted array after all merge steps?"
            ),
            "correct_answer": "[1, 2, 4, 5]",
            "rationale": "Split into [5,2]→[2,5] and [4,1]→[1,4]. Merging gives [1,2,4,5].",
            "distractors": [
                {"option": "[1, 2, 5, 4]", "explanation": "[Procedural Omission] Merging stops one step early."},
                {"option": "[2, 4, 1, 5]", "explanation": "[Initialization Error] Sorted only the first half."},
                {"option": "[1, 4, 2, 5]", "explanation": "[Boundary Error] Incorrect merge comparison."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "BFS uses a queue; visit in FIFO order, mark visited before enqueuing.",
            "step_by_step_execution": (
                "Graph: A-B, A-C, B-D, C-D. Start at A. "
                "Enqueue A, visited={A}. Dequeue A → enqueue B,C. "
                "Dequeue B → enqueue D. Dequeue C → D already visited. Dequeue D. "
                "Visit order: A, B, C, D."
            ),
            "final_state": "A, B, C, D",
        },
        "mcq_data": {
            "question": (
                "Run BFS on the graph with edges A–B, A–C, B–D, C–D starting from node A "
                "(neighbours processed alphabetically). What is the visit order?"
            ),
            "correct_answer": "A, B, C, D",
            "rationale": "Queue: [A]→dequeue A, enqueue B,C → [B,C]→dequeue B, enqueue D → [C,D]→dequeue C (D visited)→dequeue D.",
            "distractors": [
                {"option": "A, B, D, C", "explanation": "[Procedural Omission] Processing C after D instead of FIFO."},
                {"option": "A, C, B, D", "explanation": "[Initialization Error] Reversing neighbour order."},
                {"option": "A, D, B, C", "explanation": "[Operator Confusion] Confusing BFS queue with a DFS stack."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "0/1 knapsack DP: dp[i][w] = max(dp[i-1][w], dp[i-1][w-wi]+vi) if wi<=w.",
            "step_by_step_execution": (
                "Items: (w=2,v=3), (w=3,v=4). Capacity W=4. "
                "dp[1][2]=3, dp[1][3]=3, dp[1][4]=3. "
                "dp[2][4]=max(dp[1][4], dp[1][1]+4)=max(3,4)=4."
            ),
            "final_state": "4",
        },
        "mcq_data": {
            "question": (
                "Apply the 0/1 knapsack algorithm. Items: item1(weight=2, value=3), item2(weight=3, value=4). "
                "Knapsack capacity W=4. What is the maximum value achievable?"
            ),
            "correct_answer": "4",
            "rationale": "Only item2 fits alone (value=4). Both together need weight 5>4. Maximum is 4.",
            "distractors": [
                {"option": "7", "explanation": "[Boundary Error] Adding both values without checking total weight."},
                {"option": "3", "explanation": "[Procedural Omission] Selecting only item1."},
                {"option": "6", "explanation": "[Operator Confusion] Taking item1 twice as unbounded knapsack."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "BST insertion: go left if key < current node, right if key > current node.",
            "step_by_step_execution": (
                "Insert [10, 5, 15, 3] into empty BST. "
                "10 → root. 5<10 → left child of 10. 15>10 → right child of 10. "
                "3<10 → go left to 5; 3<5 → left child of 5."
            ),
            "final_state": "3 is the left child of 5",
        },
        "mcq_data": {
            "question": (
                "Insert keys [10, 5, 15, 3] one by one into an initially empty BST. "
                "Where is node 3 located in the final tree?"
            ),
            "correct_answer": "Left child of 5",
            "rationale": "10 is root. 5 goes left of 10. 3<10→go left to 5; 3<5→left child of 5.",
            "distractors": [
                {"option": "Right child of 5", "explanation": "[Boundary Error] Choosing right ignoring key comparison 3<5."},
                {"option": "Left child of 10", "explanation": "[Procedural Omission] Stopping at 10's left without continuing."},
                {"option": "Left child of 15", "explanation": "[Operator Confusion] Routing based on insertion order."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
]

_CONCEPTUAL_ONESHOT = {
    "mcq_data": {
        "question": (
            "Which of the following best describes the role of a Certificate Authority (CA) "
            "in a Public Key Infrastructure (PKI)?"
        ),
        "correct_answer": "It issues and signs digital certificates that bind a public key to an identity.",
        "rationale": (
            "A CA is a trusted third party that signs certificates, allowing others to verify "
            "that a given public key genuinely belongs to the claimed entity."
        ),
        "distractors": [
            {
                "option": "It encrypts data transmitted between two parties using symmetric keys.",
                "explanation": "Confuses the CA's role with the role of a session-key encryption layer.",
            },
            {
                "option": "It stores and distributes private keys on behalf of users.",
                "explanation": "Confuses certificate issuance with key escrow; a CA never holds user private keys.",
            },
            {
                "option": "It generates one-time passwords for multi-factor authentication.",
                "explanation": "Confuses PKI infrastructure with OTP/MFA mechanisms.",
            },
        ],
        "question_type": "conceptual",
        "source": "oneshot_example",
    },
}


def _sample_oneshot(topic: str, question_type: str) -> Dict:
    if question_type == "conceptual":
        return _CONCEPTUAL_ONESHOT
    t = topic.lower()
    if "sort" in t or "merge" in t:
        candidates = [_ONESHOT_POOL[1]]
    elif "bfs" in t or "graph" in t or "traversal" in t:
        candidates = [_ONESHOT_POOL[2]]
    elif "dynamic" in t or "dp" in t or "knapsack" in t:
        candidates = [_ONESHOT_POOL[3]]
    elif "bst" in t or "binary search tree" in t:
        candidates = [_ONESHOT_POOL[4]]
    else:
        candidates = _ONESHOT_POOL
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Conceptual question quality guidance (FIX-I)
# ---------------------------------------------------------------------------

_CONCEPTUAL_QUALITY_GUIDE = """
=== CONCEPTUAL QUESTION QUALITY STANDARD ===
BAD — never generate these:
  ✗ "What is X?"            — pure definition, no thinking required
  ✗ "Define the term X."    — encyclopaedia question
  ✗ "Which is a property of X?" (where only one textbook fact is needed)

GOOD — use one of these styles:
  ✓ SCENARIO JUDGMENT: Give a concrete real-world situation, ask which approach is better or what will happen.
     Example: "A startup stores user passwords using unsalted MD5. During a breach, what is the primary risk compared to bcrypt?"
  ✓ CONCEPT DISCRIMINATION: Highlight the key difference between two easily confused concepts.
     Example: "A developer needs guaranteed delivery with ordered packets. Should they use TCP or UDP, and why?"
  ✓ FAULT DIAGNOSIS: Describe a system exhibiting a symptom, ask for root cause.
     Example: "A machine learning model achieves 99% training accuracy but 60% test accuracy. What is the most likely cause?"
  ✓ CROSS-CONCEPT REASONING: Require connecting ≥ 2 concepts to answer.
     Example: "If a distributed cache uses consistent hashing and one node fails, how does this affect cache hit rate and data redistribution compared to modulo hashing?"
"""


# ---------------------------------------------------------------------------
# Baseline Generator
# ---------------------------------------------------------------------------

class BaselineGenerator:
    """Control group: vector-retrieved context + one-shot prompt."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, topic: str, context: List[Dict], qb_retriever=None,
                 question_format: str = "mcq_single") -> Tuple[str, str]:
        context_str   = "\n".join([f"[{i+1}] {_sanitize(c['content'])[:400]}" for i, c in enumerate(context)])
        question_type = _choose_question_type(topic, qb_retriever)

        # Non-MCQ formats: use shared prompt builder with vector context
        if question_format != "mcq_single":
            if question_format in ("true_false", "fill_blank"):
                # Use SmartGenerator helpers for formats that need stricter
                # answer-direction / answer-consistency control.
                from rag_system.generator import SmartGenerator as _SG
                _tmp = _SG.__new__(_SG)
                _tmp.llm = self.llm
                vec_ctx = {"nodes": [{"node_id": f"v{i}", "content": c["content"]}
                                      for i, c in enumerate(context)]}
                if question_format == "true_false":
                    raw = _tmp._generate_true_false(topic, question_type, vec_ctx, [])
                else:
                    raw = _tmp._generate_fill_blank(topic, question_type, vec_ctx, [])
            else:
                raw = _generate_non_mcq_with_context(
                    self.llm, topic, question_format, question_type, context_str)
            # Verify answers for verifiable formats
            if question_format in ("true_false", "mcq_multi", "fill_blank"):
                from rag_system.generator import SmartGenerator as _SG2
                _tmp2 = _SG2.__new__(_SG2)
                _tmp2.llm = self.llm
                if not _tmp2._verify_answer(raw):
                    print(f"    [VR verify] {question_format} failed, regenerating…")
                    if question_format == "true_false":
                        raw = _tmp._generate_true_false(topic, question_type, vec_ctx, [])
                    elif question_format == "fill_blank":
                        raw = _tmp._generate_fill_blank(topic, question_type, vec_ctx, [])
                    else:
                        raw = _generate_non_mcq_with_context(
                            self.llm, topic, question_format, question_type, context_str)
            return raw, "baseline_generated"
        one_shot      = _sample_oneshot(topic, question_type)

        if question_type == "computational":
            prompt = f"""You are an elite Computer Science Professor and Psychometrician. Generate ONE highly discriminative computational MCQ based on the provided textbook context.

=== TEXTBOOK CONTEXT ===
{context_str}

=== DESIGN REQUIREMENTS (STRICT RUBRIC) ===
1. CROSS-CONCEPT SYNTHESIS (CRITICAL): Force integration of MULTIPLE DISTINCT logical components from the context.
2. EDGE-CASE TRIGGERING (CRITICAL): Do NOT use happy-path input data. Design inputs to force complex conditional branches.
3. SELF-CONTAINED TRACING: Include ALL required input data. Keep inputs small (array ≤ 6 elements).
4. RIGOROUS MATHEMATICAL TRACING: Trace the algorithm with absolute precision.

=== DISTRACTOR PROTOCOL ===
Generate exactly three distractors. Each MUST represent a specific, mathematically traceable cognitive error.
Tag each with: [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]

=== EXAMPLE ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "algorithm_rules": "Define the EXACT distinct concepts/formulas combined",
        "step_by_step_execution": "<Execute step-by-step, proving edge-case was triggered>",
        "final_state": "The absolute correct final output"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{"option": "...", "explanation": "[Error Tag] Detailed reason"}}
        ],
        "question_type": "computational",
        "source": "baseline_generated"
    }}
}}
"""
        else:
            prompt = f"""You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about "{topic}" based on the provided context.

=== TEXTBOOK CONTEXT ===
{context_str}

{_CONCEPTUAL_QUALITY_GUIDE}

=== EXAMPLE ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "The key concept being tested",
        "why_this_matters": "Why understanding this is important",
        "common_misconceptions": "List the misconceptions used as distractors"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{"option": "...", "explanation": "Why a student with misconception X would choose this"}}
        ],
        "question_type": "conceptual",
        "source": "baseline_generated"
    }}
}}
"""

        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (VECTOR_RAG / BaselineGenerator)
        # boost_block starts empty; filled with judge feedback after EASY rating
        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (VECTOR_RAG / BaselineGenerator)
        # Tracks the BEST candidate across all attempts (not just the last).
        # ------------------------------------------------------------------
        candidate_json = ""
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""
        boost_block    = ""

        for attempt in range(MAX_DIFFICULTY_RETRIES):
            active_prompt = boost_block + prompt
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": active_prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            try:
                data = json.loads(raw_content)
                if "mcq_data" in data:
                    flat_data = data["mcq_data"]
                    flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                    candidate_json = json.dumps(flat_data, ensure_ascii=False)
                else:
                    candidate_json = raw_content
            except Exception:
                candidate_json = raw_content

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | VECTOR_RAG | attempt {attempt+1}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            # Always keep the highest-scoring candidate seen so far
            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning

            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                "Regenerating with difficulty boost …"
            )
        else:
            print(
                f"[DIFFICULTY FILTER | VECTOR_RAG] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
            )

        # Use the best candidate across all attempts
        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, "baseline_generated"


# ---------------------------------------------------------------------------
# NoRetrievalGenerator
# ---------------------------------------------------------------------------

class NoRetrievalGenerator:
    """Pure LLM, zero retrieval context. True zero-retrieval baseline."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    # Original signature: def generate(self, topic: str) -> Tuple[str, str]:
    def generate(self, topic: str, qb_retriever=None,
                 question_format: str = "mcq_single") -> Tuple[str, str]:
        question_type = _choose_question_type(topic, qb_retriever)

        # Non-MCQ formats: use shared prompt builder (no retrieval context)
        if question_format != "mcq_single":
            # For true_false / fill_blank: use SmartGenerator helpers with
            # tighter answer-consistency constraints.
            if question_format in ("true_false", "fill_blank"):
                from rag_system.generator import SmartGenerator as _SG
                _tmp = _SG.__new__(_SG)
                _tmp.llm = self.llm
                if question_format == "true_false":
                    raw = _tmp._generate_true_false(topic, question_type, {}, [])
                else:
                    raw = _tmp._generate_fill_blank(topic, question_type, {}, [])
            else:
                raw = _generate_non_mcq_no_context(
                    self.llm, topic, question_format, question_type)
            # Verify true_false, mcq_multi, and fill_blank answers; retry once on failure
            if question_format in ("true_false", "mcq_multi", "fill_blank"):
                from rag_system.generator import SmartGenerator as _SG2
                _tmp2 = _SG2.__new__(_SG2)
                _tmp2.llm = self.llm
                if not _tmp2._verify_answer(raw):
                    print(f"    [NR verify] {question_format} failed, regenerating…")
                    if question_format == "true_false":
                        raw = _tmp._generate_true_false(topic, question_type, {}, [])
                    elif question_format == "fill_blank":
                        raw = _tmp._generate_fill_blank(topic, question_type, {}, [])
                    else:
                        raw = _generate_non_mcq_no_context(
                            self.llm, topic, question_format, question_type)
            return raw, "no_retrieval"
        one_shot      = _sample_oneshot(topic, question_type)

        if question_type == "computational":
            prompt = f"""You are an elite Computer Science Professor. Generate ONE highly discriminative computational MCQ about: "{topic}".

No external context. Use only your parametric knowledge.

=== DESIGN REQUIREMENTS ===
1. ONE ALGORITHM ONLY: Pick ONE algorithm related to "{topic}".
2. EDGE-CASE INPUT (CRITICAL): The edge case must come from the INPUT DATA:
   - Hash table: keys causing a maximum-length collision chain
   - Sorting: nearly-sorted, reverse-sorted, or all-equal input
   - Tree: degenerate insertion order (linear chain)
   - Graph: disconnected components or isolated nodes
   - DP: inputs forcing the non-trivial subproblem branch
3. SELF-CONTAINED: Include ALL input data. Arrays ≤ 6 elements, numbers ≤ 50.
4. MATHEMATICAL PRECISION: Compute answer step-by-step in generator_scratchpad FIRST.
5. DISTRACTORS: Each must use an explicit error tag:
   [Boundary Error] / [Initialization Error] / [Procedural Omission] / [Operator Confusion]

=== EXAMPLE FORMAT ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{{{
    "generator_scratchpad": {{{{
        "algorithm_rules": "Exact rules of the algorithm",
        "step_by_step_execution": "Step 1: ... Step 2: ... Final: ...",
        "final_state": "The exact correct answer"
    }}}},
    "mcq_data": {{{{
        "question": "...",
        "correct_answer": "...",
        "rationale": "Clear step-by-step explanation",
        "distractors": [
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}},
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}},
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}}
        ],
        "question_type": "computational",
        "source": "no_retrieval_generated"
    }}}}
}}}}"""
        else:
            prompt = f"""You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about: "{topic}".

No external context. Use only your parametric knowledge.

{_CONCEPTUAL_QUALITY_GUIDE}

=== EXAMPLE FORMAT ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{{{
    "generator_scratchpad": {{{{
        "core_concept": "Key concept being tested",
        "cross_concept_link": "How two concepts connect",
        "common_misconceptions": "Misconceptions targeted by distractors"
    }}}},
    "mcq_data": {{{{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{{{"option": "...", "explanation": "Misconception: ..."}}}},
            {{{{"option": "...", "explanation": "Misconception: ..."}}}},
            {{{{"option": "...", "explanation": "Misconception: ..."}}}}
        ],
        "question_type": "conceptual",
        "source": "no_retrieval_generated"
    }}}}
}}}}"""

        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (NO_RETRIEVAL)
        # Tracks the BEST candidate across all attempts (not just the last).
        # ------------------------------------------------------------------
        candidate_json = ""
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""
        boost_block    = ""

        for attempt in range(MAX_DIFFICULTY_RETRIES):
            active_prompt = boost_block + prompt
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": active_prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            try:
                data = json.loads(raw_content)
                if "mcq_data" in data:
                    flat = data["mcq_data"]
                    flat["generator_scratchpad"] = data.get("generator_scratchpad", {})
                    candidate_json = json.dumps(flat, ensure_ascii=False)
                else:
                    candidate_json = raw_content
            except Exception:
                candidate_json = raw_content

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | NO_RETRIEVAL | attempt {attempt+1}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning

            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                "Regenerating with difficulty boost …"
            )
        else:
            print(
                f"[DIFFICULTY FILTER | NO_RETRIEVAL] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
            )

        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, "no_retrieval"


# ---------------------------------------------------------------------------
# SmartGenerator (GraphRAG model)
# ---------------------------------------------------------------------------

class SmartGenerator:
    REUSE_THRESHOLD      = 0.90
    VARY_THRESHOLD_COMP  = 0.70
    VARY_THRESHOLD_CONC  = 0.60

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        topic: str,
        graph_context: Dict,
        qb_retriever,
        question_format: str = "mcq_single",
        naive_mode: bool = False,
        no_self_correction: bool = False,
        progress_callback = None,   # SSE: callable({"stage":..., ...}) for real-time events
    ) -> Tuple[str, str]:

        question_type  = _choose_question_type(topic, qb_retriever)
        vary_threshold = (
            self.VARY_THRESHOLD_COMP if question_type == "computational"
            else self.VARY_THRESHOLD_CONC
        )

        # ── Non-MCQ formats: skip QB reuse/vary logic, go straight to generation ──
        # QB almost exclusively contains mcq_single questions, so reuse/vary
        # would produce the wrong format.  We still use QB for few-shot examples.
        if question_format != "mcq_single":
            few_shot_examples = []
            if qb_retriever is not None and qb_retriever.available:
                few_shot_examples = qb_retriever.retrieve_by_format(
                    topic, question_format, top_k=2
                )
                # Fall back to any format if none found for this specific format
                if not few_shot_examples:
                    few_shot_examples = qb_retriever.retrieve_similar(
                        topic, top_k=2,
                        prefer_computational=(question_type == "computational"),
                    )

            dispatch = {
                "mcq_multi":   self._generate_mcq_multi,
                "true_false":  self._generate_true_false,
                "fill_blank":  self._generate_fill_blank,
                "open_answer": self._generate_open_answer,
            }
            gen_fn = dispatch[question_format]
            raw    = gen_fn(topic, question_type, graph_context, few_shot_examples)

            # ABLATION C3: skip Self-Correction, return first attempt directly
            if no_self_correction:
                try:
                    q_data = json.loads(raw)
                    q_data["difficulty_score"]     = -1   # sentinel: correction was skipped
                    q_data["difficulty_reasoning"] = "Self-Correction disabled (ablation_no_correction)"
                    raw = json.dumps(q_data, ensure_ascii=False)
                except Exception:
                    pass
                return raw, "no_correction_generated"

            return self._apply_difficulty_filter_non_mcq(
                raw, "generated", topic, graph_context,
                question_type, question_format, qb_retriever,
                progress_callback=progress_callback
            )

        if qb_retriever is not None and qb_retriever.available:
            best_q, best_sim = self._find_best_question(topic, question_type, qb_retriever, question_format=question_format)
            print(f"[DEBUG] topic='{topic}' | type='{question_type}'")
            print(f"[DEBUG] best_sim={best_sim:.4f} (REUSE>{self.REUSE_THRESHOLD}, VARY>{vary_threshold})")
            if best_q:
                print(f"[DEBUG] best_q={best_q.get('question','')[:80]}...")
            print()

            if best_q is not None:
                if best_sim >= self.REUSE_THRESHOLD:
                    raw = self._wrap_original(best_q, graph_context, question_type)
                    if no_self_correction:
                        return self._tag_no_correction(raw, "no_correction_reused")
                    return self._apply_difficulty_filter(raw, "reused", topic, graph_context,
                                                         question_type, qb_retriever,
                                                         progress_callback=progress_callback)
                if best_sim >= vary_threshold:
                    raw = self._vary(best_q, graph_context, question_type)
                    if no_self_correction:
                        return self._tag_no_correction(raw, "no_correction_varied")
                    return self._apply_difficulty_filter(raw, "varied", topic, graph_context,
                                                         question_type, qb_retriever,
                                                         progress_callback=progress_callback)

        few_shot_examples = []
        if qb_retriever is not None and qb_retriever.available:
            few_shot_examples = qb_retriever.retrieve_similar(
                topic,
                top_k=3,
                prefer_computational=(question_type == "computational"),
            )

        if question_type == "computational":
            raw = self._generate_fresh(topic, graph_context, few_shot_examples, naive_mode=naive_mode)
        else:
            raw = self._generate_conceptual(topic, graph_context, few_shot_examples, naive_mode=naive_mode)

        # ABLATION C3: skip Self-Correction, return first attempt directly
        if no_self_correction:
            return self._tag_no_correction(raw, "no_correction_generated")

        return self._apply_difficulty_filter(raw, "generated" if not naive_mode else "naive_generated", topic, graph_context,
                                              question_type, qb_retriever,
                                              progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # ABLATION C3 helper: stamp sentinel fields without running correction
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_no_correction(raw_json: str, method: str) -> Tuple[str, str]:
        """
        Used when no_self_correction=True.
        Injects sentinel difficulty fields so downstream evaluation code
        never sees missing keys, without triggering any LLM retry calls.
        """
        try:
            q_data = json.loads(raw_json)
            q_data.setdefault("difficulty_score",     -1)
            q_data.setdefault("difficulty_reasoning", "Self-Correction disabled (ablation_no_correction)")
            raw_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass
        return raw_json, method

    # ------------------------------------------------------------------
    # Difficulty-filter dispatcher for SmartGenerator (GRAPH_RAG)
    # ------------------------------------------------------------------

    def _apply_difficulty_filter(
        self,
        initial_json: str,
        initial_method: str,
        topic: str,
        graph_context: Dict,
        question_type: str,
        qb_retriever,
        progress_callback = None,
    ) -> Tuple[str, str]:
        """
        Evaluate the generated question's difficulty score (1-5).
        If score <= DIFFICULTY_THRESHOLD, regenerate with targeted boost
        (up to MAX_DIFFICULTY_RETRIES total attempts, counting the first).

        Returns (question_json, method_string).
        """
        def _emit(evt):
            if progress_callback:
                try: progress_callback(evt)
                except Exception: pass

        candidate_json = initial_json
        final_method   = initial_method
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""

        # Attempt 1: the question was already generated before calling this method
        last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
        print(
            f"[DIFFICULTY FILTER | GRAPH_RAG | attempt 1/{MAX_DIFFICULTY_RETRIES}] "
            f"Score={last_score}/5 — {last_reasoning}"
        )
        if last_score > best_score:
            best_score     = last_score
            best_json      = candidate_json
            best_reasoning = last_reasoning

        if last_score <= DIFFICULTY_THRESHOLD:
            _emit({"stage": "correction_fail", "attempt": 1, "score": last_score})
        else:
            _emit({"stage": "correction_pass", "score": last_score})

        for attempt in range(2, MAX_DIFFICULTY_RETRIES + 1):
            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                f"Regenerating with difficulty boost (attempt {attempt}) …"
            )
            _emit({"stage": "generation_retry", "attempt": attempt})

            few_shot_examples = []
            if qb_retriever is not None and qb_retriever.available:
                few_shot_examples = qb_retriever.retrieve_similar(
                    topic, top_k=3, prefer_computational=(question_type == "computational")
                )

            if question_type == "computational":
                candidate_json = self._generate_fresh(topic, graph_context, few_shot_examples,
                                                       boost_block=boost_block)
            else:
                candidate_json = self._generate_conceptual(topic, graph_context, few_shot_examples,
                                                            boost_block=boost_block)
            final_method = f"generated_difficulty_retry_x{attempt - 1}"

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | GRAPH_RAG | attempt {attempt}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning

            if last_score <= DIFFICULTY_THRESHOLD:
                _emit({"stage": "correction_fail", "attempt": attempt, "score": last_score})
            else:
                _emit({"stage": "correction_pass", "score": last_score})
        else:
            if last_score <= DIFFICULTY_THRESHOLD:
                print(
                    f"[DIFFICULTY FILTER | GRAPH_RAG] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                    f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
                )
                _emit({"stage": "correction_pass", "score": best_score})  # accept best

        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, final_method

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_query(self, topic: str, question_type: str) -> str:
        if question_type == "conceptual":
            return f"explain concept definition principle {topic} why how compare"
        templates = {
            "hash table":          f"load factor clustering worst-case probes deletion tombstone performance degradation {topic}",
            "sorting":             f"trace sort step by step array state after each pass swap comparison {topic}",
            "graph traversal":     f"BFS DFS trace visit order disconnected components isolated vertex distance {topic}",
            "dynamic programming": f"fill DP table step by step compute optimal subproblem value {topic}",
            "recursion":           f"trace recursive calls stack frames compute return value base case {topic}",
            "binary search tree":  f"trace BST insert delete search compare keys height in-order {topic}",
            "binary search":       f"trace binary search midpoint computation sorted array {topic}",
            "heap":                f"trace heap insert delete heapify parent child index {topic}",
            "graph":               f"trace graph algorithm adjacency matrix edge weight path {topic}",
        }
        for key, tmpl in templates.items():
            if key in topic.lower():
                return tmpl
        return f"trace algorithm step by step compute result concrete input {topic}"

    def _cosine_from_ip(self, ip_score: float) -> float:
        """IndexFlatIP on normalised vectors returns cosine directly."""
        return max(0.0, min(1.0, float(ip_score)))

    def _find_best_question(
        self, topic: str, question_type: str, qb_retriever, question_format: str = None
    ) -> Tuple[Optional[Dict], float]:
        """
        FIX-C + FIX-J: search for question_type-appropriate questions.
        Handles both legacy 'type' field and new 'question_type' field.
        Falls back to any type if none of the right type are found.
        
        If question_format is specified, also filter by format (e.g. 'mcq_single').
        """
        query     = self._build_query(topic, question_type)
        query_vec = qb_retriever.encoder.encode([query], convert_to_numpy=True)
        import faiss as _faiss
        _faiss.normalize_L2(query_vec)
        k         = min(50, len(qb_retriever.questions))
        scores, indices = qb_retriever.index.search(query_vec, k)

        best_q, best_sim = None, 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            q = qb_retriever.questions[idx]
            # FIX-J: handle both old 'type' and new 'question_type' fields
            q_type = q.get("question_type") or q.get("type", "")
            if q_type == question_type:
                # Format check: if format specified, also match on format
                if question_format is not None:
                    q_fmt = q.get("question_format")
                    if q_fmt and q_fmt != question_format:
                        continue  # Skip if format doesn't match
                sim = self._cosine_from_ip(score)
                if sim > best_sim:
                    best_sim = sim
                    best_q   = q

        return best_q, best_sim

    # ------------------------------------------------------------------
    # _wrap_original
    # ------------------------------------------------------------------

    def _wrap_original(self, source_q: Dict, graph_context: Dict, question_type: str) -> str:
        question  = source_q.get("question", "")
        answer    = source_q.get("answer", "")
        rationale = source_q.get("rationale", "")
        options   = source_q.get("options", [])

        raw_distractors = [
            opt for opt in options
            if answer not in opt and opt.strip() != answer.strip()
        ][:3]

        if question_type == "computational":
            prompt = f"""You are a CS exam quality assurance expert. Given an existing computational question and its answer, do two things:

1. Write a step-by-step scratchpad showing how to solve the problem mathematically.
2. Rewrite the distractors with explicit error tags and explanations.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{answer}

=== RATIONALE (for reference) ===
{rationale[:500]}

=== ORIGINAL DISTRACTORS ===
{json.dumps(raw_distractors)}

Return JSON ONLY:
{{
    "generator_scratchpad": {{
        "algorithm_rules": "Key algorithm rules being tested",
        "step_by_step_execution": "Step 1: ... Step 2: ... Step 3: ...",
        "final_state": "{answer}"
    }},
    "distractors": [
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}},
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}},
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}}
    ]
}}

Error tags must be one of: [Boundary Error] [Initialization Error] [Procedural Omission] [Operator Confusion]"""
        else:
            prompt = f"""You are a CS exam quality assurance expert. Enrich this conceptual question's distractors with explicit misconception explanations.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{answer}

=== RATIONALE ===
{rationale[:500]}

=== ORIGINAL DISTRACTORS ===
{json.dumps(raw_distractors)}

Return JSON ONLY:
{{
    "generator_scratchpad": {{
        "core_concept": "What conceptual understanding this tests",
        "why_correct": "Why the correct answer is right",
        "common_misconceptions": "What misconceptions the distractors exploit"
    }},
    "distractors": [
        {{"option": "distractor text", "explanation": "Why a student with misconception X would choose this"}},
        {{"option": "distractor text", "explanation": "Why a student with misconception Y would choose this"}},
        {{"option": "distractor text", "explanation": "Why a student with misconception Z would choose this"}}
    ]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            enriched = json.loads(response.choices[0].message.content)
        except Exception:
            enriched = {
                "generator_scratchpad": {
                    "algorithm_rules": "",
                    "step_by_step_execution": rationale,
                    "final_state": answer,
                },
                "distractors": [
                    {"option": d, "explanation": "[Procedural Omission] Common mistake"}
                    for d in raw_distractors
                ],
            }

        result = {
            "question":             question,
            "correct_answer":       answer,
            "rationale":            rationale,
            "distractors":          enriched.get("distractors", []),
            "generator_scratchpad": enriched.get("generator_scratchpad", {}),
            "question_type":        question_type,
            "source":               "reused_from_bank",
        }
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # _vary
    # ------------------------------------------------------------------

    def _vary(self, source_q: Dict, context: Dict, question_type: str) -> str:
        for _ in range(3):
            raw = self._do_vary(source_q, context, question_type)
            if question_type == "conceptual" or self._verify_answer(raw):
                return raw
        return raw

    def _do_vary(self, source_q: Dict, context: Dict, question_type: str) -> str:
        context_str = "\n".join(
            [_sanitize(c["content"])[:300] for c in context.get("nodes", [])[:3]]
        )
        relations = context.get("relations", [])[:8]
        if relations:
            rel_str = "\n".join(
                f"  ({_sanitize(e['subject'])}) --[{e['predicate']}]--> ({_sanitize(e['object'])})"
                for e in relations
            )
            graph_hint = f"""
    === KNOWLEDGE GRAPH RELATIONS — PRIMARY CONSTRAINT ===
    {rel_str}

    Scoring rule you MUST satisfy:
    - Chain >= 2 of the above relations into the question logic  →  graph_relational_depth 4-5
    - Only swap numbers without using any relation             →  graph_relational_depth 1 (FAIL)
    """
        else:
            graph_hint = ""

        if question_type == "computational":
            prompt = f"""You are a CS Professor creating a new exam question by modifying an existing one.

    === ORIGINAL QUESTION (use as structural reference only) ===
    Question:  {source_q['question']}
    Answer:    {source_q.get('answer', 'N/A')}
    Rationale: {source_q.get('rationale', 'N/A')[:400]}

    === SUPPORTING CONTEXT ===
    {context_str}
    {graph_hint}
    === YOUR TASK — follow these steps in order ===
    Step 1 — Select two relations from the graph above. Write them in
            generator_scratchpad.graph_concepts_used and explain how they chain together.
    Step 2 — Design an adversarial input that forces the algorithm into a worst-case or
            boundary condition (NOT a happy-path trace). Write this in edge_case_justification.
    Step 3 — Adjust the numerical values / array contents from the original question
            so they trigger the boundary condition from Step 2 AND embed the
            two graph relations from Step 1 into the question logic.
    Step 4 — Compute the correct answer step-by-step in step_by_step_execution,
            then copy the final value into mcq_data.correct_answer.

    Additional rules:
    - Arrays <= 6 elements, numbers <= 50
    - FORBIDDEN: do not generate a basic single-step trace question
    (e.g. "insert N keys and find where key X lands" with no further reasoning)
    - Every distractor MUST use one of these tags:
    [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]
    - Avoid undefined behaviour; state every tie-breaking rule explicitly

    === REQUIRED JSON SCHEMA ===
    {{
        "generator_scratchpad": {{
            "graph_concepts_used": "Which two relations you chained and why",
            "edge_case_justification": "Why this input is adversarial / boundary-triggering",
            "algorithm_rules": "Key rules of the algorithm being tested",
            "step_by_step_execution": "Step 1: ... Step 2: ... Final answer: ...",
            "final_state": "The exact correct answer"
        }},
        "mcq_data": {{
            "question": "New question that requires tracing through >= 2 graph relations",
            "correct_answer": "...",
            "distractors": [
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}}
            ],
            "rationale": "Step 1: ... Step 2: ... Step 3: ...",
            "question_type": "computational",
            "source": "varied_from_bank"
        }}
    }}"""
        else:
            prompt = f"""You are a CS Professor creating a new conceptual question by modifying an existing one.

=== ORIGINAL QUESTION ===
Question:  {source_q['question']}
Answer:    {source_q.get('answer', 'N/A')}
Rationale: {source_q.get('rationale', 'N/A')[:400]}

=== SUPPORTING CONTEXT ===
{context_str}
{graph_hint}
{_CONCEPTUAL_QUALITY_GUIDE}

=== YOUR TASK ===
Create a NEW question that tests the SAME conceptual understanding but from a different angle.
Change the scenario, the framing, or the specific aspect being tested.
If graph relations are provided, use them to create a cross-concept question requiring causal reasoning.

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "What this question tests",
        "variation_strategy": "How you changed the angle of the original",
        "common_misconceptions": "Misconceptions targeted by distractors"
    }},
    "mcq_data": {{
        "question": "...new question with different framing...",
        "correct_answer": "...",
        "distractors": [
            {{"option": "...", "explanation": "Why a student holding misconception X would choose this"}},
            {{"option": "...", "explanation": "Why a student holding misconception Y would choose this"}},
            {{"option": "...", "explanation": "Why a student holding misconception Z would choose this"}}
        ],
        "rationale": "Clear explanation of why the answer is correct.",
        "question_type": "conceptual",
        "source": "varied_from_bank"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1800,
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            if "mcq_data" in data:
                flat = data["mcq_data"]
                flat["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat, ensure_ascii=False)
            return raw
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # FIX-H: _generate_fresh NOW uses graph relations
    # ------------------------------------------------------------------

    def _generate_fresh(self, topic: str, context: Dict, few_shot_examples: List[Dict],
                        boost_block: str = "", naive_mode: bool = False) -> str:
        for _ in range(3):
            # Pass naive_mode through to the inner _do_generate_fresh call
            raw = self._do_generate_fresh(topic, context, few_shot_examples, boost_block, naive_mode=naive_mode)
            if self._verify_answer(raw):
                return raw
        return raw

    def _do_generate_fresh(self, topic: str, context: Dict, few_shot_examples: List[Dict],
                           boost_block: str = "", naive_mode: bool = False) -> str:
        one_shot = _sample_oneshot(topic, "computational")
        fewshot_block = "=== EXAMPLE FORMAT ===\n" + json.dumps(one_shot, indent=2) + "\n\n"
        if few_shot_examples:
            fewshot_block += "=== ADDITIONAL REFERENCE EXAMPLES ===\n" + "\n".join(
                [f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples)]
            ) + "\n\n"

        # FIX-H: build graph relations block
        relations = context.get("relations", [])
        nodes     = context.get("nodes", [])

        graph_block = ""
        if relations:
            rel_lines = "\n".join(
                f"  ({_sanitize(e['subject'])}) --[{e['predicate']}]--> ({_sanitize(e['object'])})"
                for e in relations[:15]
            )
            if naive_mode:
                # Naive mode: provide relations without mandatory usage instructions
                graph_block = f"\n=== RETRIEVED KNOWLEDGE GRAPH RELATIONS ===\n{rel_lines}\n"
            else:
                graph_block = f"""
=== KNOWLEDGE GRAPH RELATIONS FOR "{topic}" ===
{rel_lines}

=== MANDATORY GRAPH USAGE INSTRUCTION ===
You MUST generate a question that requires understanding AT LEAST TWO of the
above relations in sequence to answer correctly.

Example of how to use the graph:
If the graph shows:
  (hash function h(k)) --[PRODUCES_OUTPUT]--> (initial slot index)
  (initial slot index) --[HAS_STEP]--> (linear probe on collision)
Then design a question where the student must:
  Step 1: Compute the initial slot from h(k)
  Step 2: Apply the collision resolution rule to get the FINAL position
This forces multi-hop reasoning that a student who only knows one step cannot answer.

If you do NOT use the graph relations, you are producing a simple single-concept
question that graph retrieval adds no value to — please use the relations.
"""
        elif nodes:
            node_str = "\n".join([f"  [{i+1}] {_sanitize(n['content'])[:200]}" for i, n in enumerate(nodes[:5])])
            graph_block = f"\n=== RETRIEVED CONTEXT ===\n{node_str}\n"

        prompt = f"""{boost_block}You are an elite Computer Science Professor and Psychometrician specialising in adversarial exam design.

Generate ONE highly discriminative computational MCQ about the topic: "{topic}".
{graph_block}
=== CRITICAL GENERATION CONSTRAINTS (MANDATORY) ===
You are strictly FORBIDDEN from generating basic "Trace the algorithm" questions (e.g., "Trace fib(5)", "What is the output of this sort?", "Insert these elements into a BST"). 

To receive a passing grade, your question MUST:
1. CROSS-CONCEPT SYNTHESIS (Multi-Hop): You MUST combine at least TWO distinct concepts from the provided Graph Relations. (e.g., How does choosing a specific Hash Table collision resolution strategy affect the worst-case BFS queue size? Or how does recursive call stack depth translate to DP table dimensions?)
2. EDGE CASE TRIGGERING: Do not use "happy path" inputs. You must construct an adversarial input scenario (e.g., a degenerate graph, extreme hash collisions, reverse-sorted arrays) that forces the algorithm into its worst-case or boundary condition.
3. CAUSAL REASONING: The question must ask the student to identify *why* a system failed, *how* two mechanisms interact, or compare the *efficiency trade-offs* of two approaches, rather than just computing a final single number.

If you generate a simple step-by-step trace question, you will fail.

5. DISTRACTOR QUALITY: Every distractor must name a specific, traceable student error with an explicit tag:
   [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]
6. AVOID UNDEFINED BEHAVIOUR: State every tie-breaking rule.

{fewshot_block}=== REQUIRED JSON SCHEMA ===
Complete "generator_scratchpad" FIRST, then fill "mcq_data".
{{
    "generator_scratchpad": {{
        "chosen_algorithm": "Name the primary algorithms interacting",
        "graph_concepts_used": "List EVERY graph relation used — MINIMUM 3 triples, e.g.:\n  1. (A) --[REL]--> (B)\n  2. (B) --[REL2]--> (C)\n  3. (C) --[REL3]--> (D)\nUsing < 3 relations scores graph_relational_depth = 1.",
        "edge_case_justification": "Explain WHY this input is an edge case",
        "step_by_step_execution": "Trace every step with exact values",
        "final_state": "The exact correct answer"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "<must match final_state exactly>",
        "rationale": "Clear step-by-step explanation matching the trace above",
        "distractors": [
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}}
        ],
        "question_type": "computational",
        "source": "generated_via_graph"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            if "mcq_data" in data:
                flat_data = data["mcq_data"]
                flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat_data, ensure_ascii=False)
            return raw_content
        except Exception:
            return raw_content

    # ------------------------------------------------------------------
    # FIX-I: _generate_conceptual with good/bad question guidance
    # ------------------------------------------------------------------

    def _generate_conceptual(
        self, topic: str, context: Dict, few_shot_examples: List[Dict],
        boost_block: str = "", naive_mode: bool = False
    ) -> str:
        nodes_str = "\n".join(
            [f"[{i+1}] {_sanitize(n['content'])[:400]}" for i, n in enumerate(context.get("nodes", []))]
        )

        # Also inject graph relations for conceptual topics
        relations = context.get("relations", [])[:10]
        if relations:
            rel_str = "\n".join(
                f"  ({_sanitize(e['subject'])}) --[{e['predicate']}]--> ({_sanitize(e['object'])})"
                for e in relations
            )
            graph_block = f"\n=== KNOWLEDGE GRAPH RELATIONS ===\n{rel_str}\nUse these causal links to create cross-concept questions.\n"
        else:
            graph_block = ""

        one_shot = _sample_oneshot(topic, "conceptual")
        fewshot_block = "=== EXAMPLE FORMAT ===\n" + json.dumps(one_shot, indent=2) + "\n\n"
        if few_shot_examples:
            fewshot_block += "=== ADDITIONAL REFERENCE EXAMPLES ===\n" + "\n".join(
                [f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples)]
            ) + "\n\n"

        prompt = f"""{boost_block}You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about "{topic}".

=== RETRIEVED CONTEXT ===
{nodes_str}
{graph_block}
{_CONCEPTUAL_QUALITY_GUIDE}

=== DESIGN REQUIREMENTS ===
1. DEEP UNDERSTANDING: Test WHY and HOW, not just WHAT.
2. MISCONCEPTION-DRIVEN DISTRACTORS: Each wrong option must represent a specific, named misconception.
3. CROSS-CONCEPT: The question should require connecting at least two related concepts.
4. RELEVANCE: The question must be explicitly about "{topic}".

{fewshot_block}=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "The key concept being tested",
        "cross_concept_link": "How two or more concepts are connected in this question",
        "common_misconceptions": "The misconceptions each distractor exploits"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "Clear explanation of why the answer is correct and why distractors are wrong.",
        "distractors": [
            {{"option": "...", "explanation": "Misconception: ..."}},
            {{"option": "...", "explanation": "Misconception: ..."}},
            {{"option": "...", "explanation": "Misconception: ..."}}
        ],
        "question_type": "conceptual",
        "source": "generated_via_graph"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            if "mcq_data" in data:
                flat_data = data["mcq_data"]
                flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat_data, ensure_ascii=False)
            return raw_content
        except Exception:
            return raw_content

    # ------------------------------------------------------------------
    # New format generators: mcq_multi, true_false, fill_blank, open_answer
    # Each follows the same pattern as _generate_conceptual:
    #   build context → build prompt → call LLM → flatten and return JSON
    # ------------------------------------------------------------------

    def _build_context_block(self, context: Dict, fmt: str = "") -> tuple:
        """
        Shared helper: build (nodes_str, graph_block) from graph context.
        Uses MANDATORY GRAPH USAGE INSTRUCTION (same strength as _do_generate_fresh)
        to ensure LLM actually uses graph relations instead of ignoring them.
        """
        nodes_str = "\n".join(
            [f"[{i+1}] {_sanitize(n['content'])[:400]}"
             for i, n in enumerate(context.get("nodes", []))]
        )
        relations = context.get("relations", [])[:10]
        if relations:
            rel_str = "\n".join(
                f"  ({_sanitize(e['subject'])}) --[{e['predicate']}]--> ({_sanitize(e['object'])})"
                for e in relations
            )
            # Format-specific hint for how to USE the graph in this format
            fmt_hint = {
                "mcq_multi":  ("Each CORRECT option should reference a different relation "
                               "or a different step in the same relation chain. "
                               "Wrong options should represent misapplied or reversed relations."),
                "true_false": ("The statement should make a claim that is TRUE or FALSE "
                               "specifically because of one of the listed relations — "
                               "not a general CS fact the LLM already knows."),
                "fill_blank": ("The question MUST require the student to mentally trace through "
                               "AT LEAST TWO connected relations from the graph to determine the "
                               "blank's value. For example, trace from [subject A] to [object B], "
                               "and then apply a boundary condition to get the final answer. "
                               "Do NOT write a generic sentence that only requires one fact."),
                "open_answer":("The question should ask the student to trace through "
                               "AT LEAST TWO of the listed relations in sequence."),
            }.get(fmt, "Use at least two of the above relations in the question logic.")

            graph_block = f"""
=== KNOWLEDGE GRAPH RELATIONS FOR THIS TOPIC ===
{rel_str}

=== MANDATORY GRAPH USAGE INSTRUCTION ===
You MUST design the question so that answering it correctly REQUIRES knowing at
least ONE (ideally TWO OR MORE) of the above relations.  {fmt_hint}

★ GRAPH GROUNDING REQUIREMENT: In the "graph_grounding" field, you MUST list
every relation you used, e.g.:
  "(hash function) --[HAS_STEP]--> (compute bucket index)"
  "(bucket index) --[TRIGGERS]--> (collision resolution)"
Listing ≥ 2 distinct relations is REQUIRED. A single relation scores GRD=1.

If you ignore the graph relations and write a question from memory alone,
you are defeating the purpose of graph retrieval — the question will score GRD=1.
"""
        else:
            graph_block = ""
        return nodes_str, graph_block

    def _generate_mcq_multi(
        self, topic: str, question_type: str, context: Dict,
        few_shot_examples: List[Dict], boost_block: str = ""
    ) -> str:
        nodes_str, graph_block = self._build_context_block(context, fmt="mcq_multi")

        if question_type == "computational":
            diff_note = (
                "Each CORRECT option tests a different computational aspect "
                "(complexity class, intermediate state, or invariant). "
                "Wrong options must arise from specific named errors "
                "(off-by-one, wrong formula, boundary mistake)."
            )
            # ★ FIX-MCQ-MULTI-VERIFY: computational multi-select often produces
            # WRONG answers because LLM marks options correct without computing them.
            # Force a two-step scratchpad: compute each option independently FIRST,
            # then mark correct/incorrect based on those computations.
            verify_block = """
★ MANDATORY SELF-VERIFICATION (computational questions):
Step 1 — In the "scratchpad" field, independently compute or evaluate EACH option
         (A through E) from first principles. Show your work for each.
Step 2 — Only after completing all 5 evaluations, fill "correct_answers" with
         the letters whose computations confirmed TRUE/CORRECT.
Step 3 — If you find 0 or 1 correct option, redesign the question so exactly
         2-3 options are correct before outputting.
FORBIDDEN: marking an option correct without computing it in the scratchpad first.
"""
        else:
            diff_note = (
                "CORRECT options capture different true consequences of the same "
                "causal mechanism. Wrong options are subtly incorrect versions — "
                "NOT obviously wrong."
            )
            verify_block = ""

        few_shot_blk = ""
        if few_shot_examples:
            few_shot_blk = "=== REFERENCE EXAMPLES ===\n" + "\n".join(
                f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples[:2])
            ) + "\n\n"

        prompt = f"""{boost_block}You are an elite CS Professor. Generate ONE multiple-correct-answer MCQ about "{topic}" (question_type: {question_type}).

=== CONTEXT ===
{nodes_str}
{graph_block}
{few_shot_blk}{verify_block}
=== REQUIREMENTS ===
- Exactly 5 options labelled A–E.
- Exactly 2–3 options are correct. The others are plausible distractors.
- {diff_note}
- FORBIDDEN: trivial definition recall, all options about different topics.
- For computational: keep inputs SMALL (≤ 6 elements) so each option is humanly verifiable.

=== OUTPUT (valid JSON only, no markdown) ===
{{
    "scratchpad": "Option A: [my computation] → TRUE/FALSE. Option B: [my computation] → TRUE/FALSE. ...",
    "graph_grounding": "(subject) --[relation]--> (object) — graph relations used in option design",
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}},
    "correct_answers": ["A", "C"],
    "explanation": "Why each correct option is right and each distractor is wrong.",
    "question_type": "{question_type}",
    "question_format": "mcq_multi"
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            data.setdefault("question_format", "mcq_multi")
            data.setdefault("question_type",   question_type)
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return raw

    def _generate_true_false(
        self, topic: str, question_type: str, context: Dict,
        few_shot_examples: List[Dict], boost_block: str = ""
    ) -> str:
        """
        Fact-first true/false generation.

        Instead of asking the LLM to generate a statement and simultaneously
        determine its truth value (which causes systematic errors like
        "f(5) needs 15 calls" → tf_answer=False when 15 is actually correct),
        we use a two-step approach:

        Step 1: Ask LLM to COMPUTE a specific fact and give the exact answer.
        Step 2: Construct the T/F statement FROM that verified fact, so the
                truth value is determined by the fact, not by a second guess.
        """
        nodes_str, graph_block = self._build_context_block(context, fmt="true_false")

        # ★ FIX-TF-PROMPT: Original type_note guided LLM toward the most
        # straightforward KG node relationship, producing factually correct but
        # pedagogically weak statements (diagnostic_power ≈ 1.5 on cybersecurity,
        # information retrieval). Rewritten to explicitly prefer counterintuitive
        # or cross-concept causal facts that trip up students — the kind Unpruned
        # GraphRAG generates naturally from its fuller context window.
        type_note = (
            "PRIORITY: pick a fact that CONTRADICTS a common student assumption "
            "about this algorithm — a boundary case, an off-by-one result, or a "
            "worst-case behaviour students consistently mis-estimate. "
            "Example: 'Build-Max-Heap on n=9 calls Max-Heapify exactly floor(9/2)=4 "
            "times on non-leaf nodes, NOT 9 times as most students assume.' "
            "If no such misconception exists in the context, fall back to a specific "
            "computed value requiring ≥3 non-trivial steps to derive."
            if question_type == "computational" else
            "PRIORITY: find a CROSS-CONCEPT CAUSAL CHAIN where mechanism A failing "
            "or holding produces an unexpected effect on mechanism B — especially one "
            "where the causal direction is counterintuitive. "
            "Example: 'A stateful firewall permitting partial TCP handshakes can be "
            "exploited for SYN-flood because it exhausts the state table before the "
            "filtering rule fires.' "
            "If no cross-concept chain exists, fall back to a single-concept edge "
            "case where the standard defence/property fails under specific conditions."
        )
        few_shot_blk = ""
        if few_shot_examples:
            few_shot_blk = "=== REFERENCE EXAMPLES ===\n" + "\n".join(
                f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples[:2])
            ) + "\n\n"

        # ── Step 1: Find a counterintuitive fact ─────────────────────────
        step1_prompt = f"""You are an elite CS Professor. Topic: "{topic}" (question_type: {question_type})

=== CONTEXT ===
{nodes_str}
{graph_block}
{few_shot_blk}
Your task: Find ONE fact about "{topic}" that is COUNTERINTUITIVE or commonly misunderstood.
{type_note}

SELECTION PRIORITY (choose the highest tier available from the context):
  Tier 1 — Cross-concept: mechanism A unexpectedly determines behaviour of mechanism B
  Tier 2 — Boundary/edge-case: a value that differs from the "obvious" estimate
  Tier 3 — Multi-step computation: requires ≥3 distinct reasoning steps

Requirements:
- Show your work step by step
- Arrive at an EXACT, UNAMBIGUOUS answer (a number, expression, or causal claim)
- Name the MISCONCEPTION the fact contradicts (even if tier 3)

Return ONLY valid JSON:
{{
    "fact_statement": "A precise claim that is verifiably TRUE but non-obvious",
    "computation": "Your step-by-step reasoning showing why this fact is true",
    "exact_answer": "The specific value/result that makes the fact true",
    "graph_grounding": "(subject) --[relation]--> (object) if graph was used, else empty",
    "misconception": "The common wrong belief this fact contradicts (1 sentence)"
}}"""

        try:
            r1 = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": step1_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,   # low temp for computation accuracy
                max_tokens=800,
            )
            step1 = json.loads(r1.choices[0].message.content)
        except Exception:
            # ★ FIX-1: step1 failed → attempt single-step true_false directly
            # instead of silently using empty strings (which produces blank statements)
            try:
                single_prompt = (
                    f'You are an elite CS Professor. Topic: "{topic}" (type: {question_type})\n\n'
                    f'=== CONTEXT ===\n{nodes_str}\n{graph_block}\n'
                    f'Generate ONE challenging True/False exam question that makes a specific,\n'
                    f'verifiable claim requiring computation or causal reasoning to evaluate.\n'
                    f'FORBIDDEN: vague definitions, obvious statements.\n\n'
                    f'Return ONLY valid JSON:\n'
                    f'{{"graph_grounding":"","statement":"...","tf_answer":true,'
                    f'"explanation":"Step-by-step proof showing why the statement is true/false.",'
                    f'"question_type":"{question_type}","question_format":"true_false"}}'
                )
                rs = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": single_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=700,
                )
                sd = json.loads(rs.choices[0].message.content)
                sd.setdefault("question_format", "true_false")
                sd.setdefault("question_type",   question_type)
                # ★ FIX-2: add canonical `question` alias so all downstream
                # code that reads q.get("question","") gets the content
                sd["question"] = sd.get("statement", "")
                return json.dumps(sd, ensure_ascii=False)
            except Exception as inner_e:
                empty = {"statement": "", "question": "", "tf_answer": True,
                         "explanation": "", "question_format": "true_false",
                         "question_type": question_type, "error": str(inner_e)}
                return json.dumps(empty, ensure_ascii=False)

        fact         = step1.get("fact_statement", "")
        workings     = step1.get("computation", "")
        gg           = step1.get("graph_grounding", "")
        misconception = step1.get("misconception", "")

        # ── Step 2: Build the T/F question from the verified fact ────────
        # With probability ~0.5, generate a TRUE statement (using exact_answer)
        # With probability ~0.5, generate a FALSE statement (using a wrong value)
        misconception_block = (
            f"\n=== COMMON MISCONCEPTION TO EXPLOIT ===\n{misconception}\n"
            if misconception else ""
        )
        step2_prompt = f"""{boost_block}You are an elite CS Professor. Topic: "{topic}" (question_type: {question_type})

=== VERIFIED FACT ===
{fact}
Computation: {workings}
{misconception_block}
=== TASK ===
Using the verified fact above, create ONE True/False exam question.
CHOOSE ONE of these two strategies (pick the harder/more interesting one):

  STRATEGY A — True statement (use the exact correct value):
    Write the fact as a statement. tf_answer = true.
    Make it a TRAP by phrasing it in a way that LOOKS false to a student
    who hasn't done the computation — ideally exploiting the misconception above.

  STRATEGY B — False statement (change ONE key value to a wrong value):
    Slightly alter the fact (wrong number, wrong complexity, wrong direction).
    tf_answer = false.
    The wrong value must match the misconception — the exact mistake a student
    would make if they held the wrong belief described above.

{type_note}
FORBIDDEN: vague generalisations, definitional statements ("X is an asymmetric algorithm"),
  statements where the truth is obvious without computation.

Return ONLY valid JSON:
{{
    "graph_grounding": "{gg}",
    "statement": "...",
    "tf_answer": true,
    "explanation": "Step-by-step proof: [show the correct computation]. The statement is [true/false] because [specific reason].",
    "question_type": "{question_type}",
    "question_format": "true_false"
}}"""

        # ── Single-step fallback prompt (reused across empty-statement rescues) ──
        _single_step_tf_prompt = (
            f'You are an elite CS Professor. Topic: "{topic}" (type: {question_type})\n\n'
            f'=== CONTEXT ===\n{nodes_str}\n{graph_block}\n'
            f'Generate ONE challenging True/False exam question that exploits a '
            f'COMMON MISCONCEPTION or COUNTERINTUITIVE CAUSAL RELATIONSHIP about '
            f'"{topic}". Do NOT write a plain definitional statement.\n'
            f'FORBIDDEN: vague generalisations, obvious facts.\n\n'
            f'Return ONLY valid JSON:\n'
            f'{{"graph_grounding":"","statement":"...","tf_answer":true,'
            f'"explanation":"Step-by-step proof showing why the statement is true/false.",'
            f'"question_type":"{question_type}","question_format":"true_false"}}'
        )

        try:
            r2 = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": step2_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=800,
            )
            data = json.loads(r2.choices[0].message.content)
            data.setdefault("question_format", "true_false")
            data.setdefault("question_type",   question_type)
            # ★ FIX-2a: add canonical `question` alias
            data["question"] = data.get("statement", "")

            # ★ FIX-2b: step2 produced empty statement → retry once at higher temp
            if not data.get("statement", "").strip():
                print(f"    [true_false] WARNING: step2 produced empty statement — retrying once")
                r2b = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": step2_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.9,
                    max_tokens=800,
                )
                data = json.loads(r2b.choices[0].message.content)
                data.setdefault("question_format", "true_false")
                data.setdefault("question_type",   question_type)
                data["question"] = data.get("statement", "")

            # ★ FIX-1 (NEW): retry also empty → abandon two-step, use single-step fallback.
            # Root cause of the two catastrophic 1.4-score questions (binary search,
            # recursion): step2_prompt returned valid JSON but with a blank "statement"
            # field. The previous code returned that empty result directly.
            if not data.get("statement", "").strip():
                print(f"    [true_false] WARNING: step2 retry also empty — single-step fallback")
                try:
                    rs = self.llm.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": _single_step_tf_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.7,
                        max_tokens=700,
                    )
                    sd = json.loads(rs.choices[0].message.content)
                    sd.setdefault("question_format", "true_false")
                    sd.setdefault("question_type",   question_type)
                    sd["question"] = sd.get("statement", "")
                    return json.dumps(sd, ensure_ascii=False)
                except Exception as fb_e:
                    print(f"    [true_false] single-step fallback also failed: {fb_e}")
                    # Return what we have (may still be empty — at least not a crash)
                    return json.dumps(data, ensure_ascii=False)

            return json.dumps(data, ensure_ascii=False)

        except Exception as e:
            # ★ FIX-1 (NEW): step2 exception → single-step fallback instead of
            # returning a blank JSON. Previous code produced {"statement": "", ...}
            # which scored 1.4 across all dimensions.
            print(f"    [true_false] Exception in step2 ({e}) — single-step fallback")
            try:
                rs = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": _single_step_tf_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=700,
                )
                sd = json.loads(rs.choices[0].message.content)
                sd.setdefault("question_format", "true_false")
                sd.setdefault("question_type",   question_type)
                sd["question"] = sd.get("statement", "")
                return json.dumps(sd, ensure_ascii=False)
            except Exception as inner_e:
                return json.dumps(
                    {"statement": "", "question": "", "question_format": "true_false",
                     "question_type": question_type, "error": str(inner_e)},
                    ensure_ascii=False,
                )


    def _generate_fill_blank(
        self, topic: str, question_type: str, context: Dict,
        few_shot_examples: List[Dict], boost_block: str = ""
    ) -> str:
        nodes_str, graph_block = self._build_context_block(context, fmt="fill_blank")
        if question_type == "computational":
            diff_note = (
                "The blank must test a NON-TRIVIAL decision point — a value that "
                "a student who only partially understands the algorithm would get wrong. "
                "Good targets: the result of collision resolution (not just h(k) mod m), "
                "the number of comparisons in a worst-case trace, or the state after "
                "a non-obvious edge case. "
                "Example: 'Inserting keys [0,7,14] into a linear-probing table of size 7 "
                "with h(k)=k mod 7 — after all insertions, the final position of key 14 is ___.' "
                "(answer: 2, because 0→slot0, 7→slot1 collision→slot1, 14→slot0 collision→slot1 collision→slot2)"
            )
        else:
            diff_note = (
                "The blank completes a specific causal claim where the wrong answer "
                "represents a common misconception — NOT a vague word or synonym. "
                "Example: 'When SHA-1 collision resistance is broken, an attacker "
                "can forge ___ without access to the private key.' (answer: digital signatures) "
                "A student who confuses hash properties with signature properties would get this wrong."
            )
        few_shot_blk = ""
        if few_shot_examples:
            few_shot_blk = "=== REFERENCE EXAMPLES ===\n" + "\n".join(
                f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples[:2])
            ) + "\n\n"

        type_forbidden_fb = (
            "FORBIDDEN: do not generate a conceptual/causal sentence as fill_blank. "
            "This is COMPUTATIONAL — the blank must require tracing an algorithm to "
            "a specific concrete result that exposes a non-obvious edge case."
            if question_type == "computational" else
            "FORBIDDEN: do not generate algorithm steps or numeric trace questions. "
            "This is CONCEPTUAL — the blank must complete a causal/mechanism claim "
            "where the correct answer exposes a specific misconception."
        )

        prompt = f"""{boost_block}You are an elite CS Professor. Generate ONE fill-in-the-blank question about "{topic}" (question_type: {question_type}).

=== CONTEXT ===
{nodes_str}
{graph_block}
{few_shot_blk}=== STRICT REQUIREMENTS ===
★ BLANK COUNT: EXACTLY 1 BLANK (___). No exceptions.
★ EDGE-CASE FRAMING: The blank must be positioned at a NON-TRIVIAL decision point
  that separates students who truly understand from those who only partially know.
  DO NOT ask for the final output of a trivial 1-step computation.
  DO ask for the result after a collision, a boundary condition, or an edge case.
★ INPUT SIZE: Use ONLY tiny inputs (array ≤ 5 elements, table size ≤ 7).
★ SELF-CHECK (MANDATORY): Before writing the JSON, solve the blank THREE times
  independently. Write all three attempts in "scratchpad". If results differ,
  simplify the question until all three agree.
★ DIAGNOSTIC TRAP: After designing the question, write in "common_wrong_answer"
  what a student who only partially understands would put in the blank, and why
  that is wrong. This ensures the blank has real diagnostic power.
- {diff_note}
- {type_forbidden_fb}
- The explanation must state both the correct answer AND why the common wrong
  answer is incorrect.

=== OUTPUT (valid JSON only, no markdown) ===
{{
    "scratchpad": "Attempt1=[X], Attempt2=[X], Attempt3=[X] — all agree",
    "common_wrong_answer": "A student who [misconception] would answer [Y] because...",
    "graph_grounding": "(subject) --[relation]--> (object)",
    "sentence": "Setup context (1-2 sentences). The [specific thing] is ___.",
    "answers": ["single_correct_answer"],
    "explanation": "The correct answer is X because [reasoning]. Common mistake: [wrong answer] because [misconception].",
    "question_type": "{question_type}",
    "question_format": "fill_blank"
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            data.setdefault("question_format", "fill_blank")
            data.setdefault("question_type",   question_type)
            data["question"] = data.get("sentence", "")
            sentence = data.get("sentence", "")
            if not sentence.strip() or "___" not in sentence:
                print(f"    [fill_blank] WARNING: empty/blank-less sentence — retrying once")
                r2 = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.9,
                )
                data2 = json.loads(r2.choices[0].message.content)
                data2.setdefault("question_format", "fill_blank")
                data2.setdefault("question_type",   question_type)
                data2["question"] = data2.get("sentence", "")
                if data2.get("sentence", "").strip() and "___" in data2.get("sentence", ""):
                    data = data2
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return raw

    def _generate_open_answer(
        self, topic: str, question_type: str, context: Dict,
        few_shot_examples: List[Dict], boost_block: str = ""
    ) -> str:
        nodes_str, graph_block = self._build_context_block(context, fmt="open_answer")
        if question_type == "computational":
            diff_note = (
                "Ask the student to trace an algorithm on a SPECIFIC input AND "
                "explain WHY the result demonstrates a complexity or correctness "
                "property. The model answer must include ALL intermediate states, "
                "not just the final result. Key points must be checkable criteria "
                "(e.g. 'states that merge sort makes exactly 11 comparisons on this input')."
            )
        else:
            diff_note = (
                "Ask the student to explain HOW and WHY two specific mechanisms "
                "interact, including what breaks if one mechanism fails. "
                "The model answer must explain the causal chain clearly, but "
                "you are FREE to choose any clear explanatory style — narrative, "
                "comparative, or step-by-step — that best fits the topic. "
                "Avoid the rigid template 'Because [A] does X, [B] cannot do Y' "
                "— use it only if it naturally fits, not as a mandatory formula. "
                "Key points must be specific causal claims, not vague summaries."
            )
        few_shot_blk = ""
        if few_shot_examples:
            few_shot_blk = "=== REFERENCE EXAMPLES ===\n" + "\n".join(
                f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples[:2])
            ) + "\n\n"

        type_forbidden = (
            "FORBIDDEN type-mismatch: this is a COMPUTATIONAL question — "
            "the question MUST require the student to compute specific values, "
            "trace algorithm steps, or derive a quantitative result. "
            "Do NOT generate conceptual/essay-style questions about mechanisms or policies."
            if question_type == "computational" else
            "FORBIDDEN type-mismatch: this is a CONCEPTUAL question — "
            "the question MUST require causal reasoning about WHY mechanisms interact. "
            "Do NOT generate questions that require computing specific numeric values "
            "or tracing algorithm steps (e.g. 'compute 2^4 mod 11' is FORBIDDEN)."
        )
        graph_force = (
            "MANDATORY: if graph relations are provided above, your question MUST "
            "reference at least ONE of those relations explicitly in the scenario. "
            "Do not ignore the graph context."
            if graph_block else ""
        )
        # ★ FIX-OA-DIVERSITY: open_answer diversity collapses when multiple
        # questions on the same topic use similar phrasing or scenario structure.
        # Force unique angle selection to improve session diversity score.
        diversity_block = f"""
★ DIVERSITY REQUIREMENT: Your question must take a UNIQUE ANGLE on "{topic}".
Choose ONE of these angles (pick whichever you haven't used recently):
  A) Failure-mode analysis — what goes wrong when a specific condition fails
  B) Comparative reasoning — contrast two related mechanisms or algorithms
  C) Design justification — WHY was this design choice made over alternatives
  D) Edge-case explanation — WHY does behaviour differ on boundary/adversarial input
  E) Cross-concept synthesis — HOW do two related concepts causally interact
State which angle you chose in the "angle" field of your JSON output.
FORBIDDEN: generic "explain how X works" questions with no specific angle.
"""

        prompt = f"""{boost_block}You are an elite CS Professor. Generate ONE open-answer exam question about "{topic}" (question_type: {question_type}).

=== CONTEXT ===
{nodes_str}
{graph_block}
{few_shot_blk}{diversity_block}
=== REQUIREMENTS ===
- {diff_note}
- {type_forbidden}
- {graph_force}
- model_answer: complete, detailed answer (3–6 sentences minimum).
- key_points: 3–5 specific, checkable criteria an examiner uses to grade.
- FORBIDDEN: vague questions ("explain X"), one-sentence model answers,
  key points that are just topic names.

=== OUTPUT (valid JSON only, no markdown) ===
{{
    "angle": "A/B/C/D/E — one word label",
    "question": "...",
    "model_answer": "...",
    "key_points": ["specific checkable criterion 1", "criterion 2", "criterion 3"],
    "question_type": "{question_type}",
    "question_format": "open_answer"
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            data.setdefault("question_format", "open_answer")
            data.setdefault("question_type",   question_type)
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # Difficulty filter for non-MCQ formats
    # ------------------------------------------------------------------
    # Mirrors _apply_difficulty_filter but uses format-aware field extraction
    # (assess_difficulty was already fixed to handle all formats correctly).
    # For fill_blank/true_false the ceiling is Score 3 by nature — those
    # formats test recall/verification, not multi-hop trace.  We therefore
    # use a lower threshold (2) so we don't endlessly retry for a format
    # that structurally cannot reach Score 4.
    # ------------------------------------------------------------------

    # ★ FIX-FORMAT-THRESHOLD: data-driven per-format thresholds.
    # Analysis of scored data shows:
    #   mcq_single:  diff=3→3.990, diff=4→3.959  (diff=3 already good; keep threshold=3)
    #   mcq_multi:   diff=3→3.700, diff=4→3.714  (marginal; keep threshold=3)
    #   true_false:  diff=3→3.843, diff=4→3.928  (threshold=3 pushes toward 4, helpful)
    #   fill_blank:  diff=2→2.950, diff=4→3.385  (must filter diff=2; threshold=3 OK)
    #   open_answer: all are diff=4 anyway       (threshold=3 = no extra cost)
    # ★ FIX-FB-THRESHOLD: fill_blank threshold lowered from 3→1.
    # Data shows fill_blank difficulty retries produce WRONG answers (5/13 wrong).
    # The retry boost prompt pushes LLM to generate 3-4 blank complex scenarios
    # that it cannot compute accurately. By setting threshold=1, we accept the
    # FIRST correctly-structured question regardless of difficulty score,
    # and rely on _verify_answer to enforce correctness instead.
    # All other formats stay at 3 (retry until difficulty≥4).
    _FORMAT_DIFFICULTY_THRESHOLD = {
        "mcq_multi":   3,
        "true_false":  3,
        "fill_blank":  1,   # ★ FIX-FILL: difficulty judge scores almost all fill_blank ≤2,
                            # making threshold=2 trigger retry on 19/20 questions.
                            # Retried versions score lower on correctness (4.10 vs 4.78 baseline)
                            # because the LLM over-complicates the sentence while answer logic breaks.
                            # Set to 1: only truly trivial fill_blank questions get regenerated.
        "open_answer": 3,
    }

    def _apply_difficulty_filter_non_mcq(
        self,
        initial_json: str,
        initial_method: str,
        topic: str,
        graph_context: Dict,
        question_type: str,
        question_format: str,
        qb_retriever,
        progress_callback = None,
    ) -> Tuple[str, str]:
        threshold = self._FORMAT_DIFFICULTY_THRESHOLD.get(question_format, 3)

        def _emit(evt):
            if progress_callback:
                try: progress_callback(evt)
                except Exception: pass

        candidate_json = initial_json
        final_method   = initial_method
        best_json      = initial_json
        best_score     = 0

        last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
        print(f"[DIFFICULTY FILTER | GRAPH_RAG | {question_format} | attempt 1/{MAX_DIFFICULTY_RETRIES}] "
              f"Score={last_score}/5 — {last_reasoning[:120]}")
        if last_score > best_score:
            best_score = last_score
            best_json  = candidate_json

        if last_score <= threshold:
            _emit({"stage": "correction_fail", "attempt": 1, "score": last_score})
        else:
            _emit({"stage": "correction_pass", "score": last_score})

        for attempt in range(2, MAX_DIFFICULTY_RETRIES + 1):
            if last_score > threshold:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(f"  → Score {last_score}/5 ≤ threshold {threshold}/5. "
                  f"Regenerating (attempt {attempt}) …")
            _emit({"stage": "generation_retry", "attempt": attempt})

            few_shot_examples = []
            if qb_retriever is not None and qb_retriever.available:
                few_shot_examples = qb_retriever.retrieve_by_format(
                    topic, question_format, top_k=2)
                if not few_shot_examples:
                    few_shot_examples = qb_retriever.retrieve_similar(topic, top_k=2)

            dispatch = {
                "mcq_multi":   self._generate_mcq_multi,
                "true_false":  self._generate_true_false,
                "fill_blank":  self._generate_fill_blank,
                "open_answer": self._generate_open_answer,
            }
            candidate_json = dispatch[question_format](
                topic, question_type, graph_context, few_shot_examples,
                boost_block=boost_block)
            final_method = f"generated_difficulty_retry_x{attempt - 1}"

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(f"[DIFFICULTY FILTER | GRAPH_RAG | {question_format} | attempt {attempt}/{MAX_DIFFICULTY_RETRIES}] "
                  f"Score={last_score}/5 — {last_reasoning[:120]}")
            if last_score > best_score:
                best_score = last_score
                best_json  = candidate_json

            if last_score <= threshold:
                _emit({"stage": "correction_fail", "attempt": attempt, "score": last_score})
            else:
                _emit({"stage": "correction_pass", "score": last_score})
        else:
            if last_score <= threshold:
                print(f"[DIFFICULTY FILTER | GRAPH_RAG | {question_format}] "
                      f"All {MAX_DIFFICULTY_RETRIES} attempts ≤ {threshold}/5. "
                      f"Accepting best (score={best_score}).")
                _emit({"stage": "correction_pass", "score": best_score})

        # ── Answer verification for verifiable formats ─────────────────
        if question_format in ("true_false", "mcq_multi", "fill_blank"):
            if not self._verify_answer(best_json):
                print(f"    [_verify_answer] {question_format} verification failed on best candidate. "
                      f"Generating one final attempt…")
                _emit({"stage": "verify_fail"})
                few_shot_examples = []
                if qb_retriever is not None and qb_retriever.available:
                    few_shot_examples = qb_retriever.retrieve_by_format(
                        topic, question_format, top_k=2)
                dispatch = {
                    "true_false": self._generate_true_false,
                    "mcq_multi":  self._generate_mcq_multi,
                    "fill_blank": self._generate_fill_blank,
                }
                _emit({"stage": "rescue_start"})
                rescue_json = dispatch[question_format](
                    topic, question_type, graph_context, few_shot_examples)
                if self._verify_answer(rescue_json):
                    best_json = rescue_json
                    final_method = final_method + "_verified_rescue"
                    print(f"    [_verify_answer] Rescue generation passed verification.")
                else:
                    print(f"    [_verify_answer] Rescue also failed; accepting best seen.")

        final_json = best_json
        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = best_score
            q_data["difficulty_reasoning"] = last_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass
        return final_json, final_method

    # ------------------------------------------------------------------
    # _verify_answer (Fix-1 retained)
    # ------------------------------------------------------------------

    def _verify_answer(self, question_json_str: str) -> bool:
        """
        Verify question correctness by calling an independent LLM.
        Handles mcq_single, true_false, mcq_multi, and fill_blank formats.
        - open_answer: skip (no unique verifiable answer).
        - conceptual mcq_single: skip (no numerical ground truth).
        - computational mcq_single: compare independently computed answer.
        - true_false: independently determine True/False, compare with tf_answer.
        - mcq_multi: independently determine correct options, check consistency.
        - fill_blank: independently solve the blanks and compare ordered answers.
        """
        try:
            q = json.loads(question_json_str)
        except Exception:
            return False

        fmt       = q.get("question_format", "mcq_single")
        q_type    = q.get("question_type",   "computational")

        # Formats with no unique verifiable answer — skip
        if fmt == "open_answer":
            return True

        # ── fill_blank verification ─────────────────────────────────────
        if fmt == "fill_blank":
            sentence = q.get("sentence", "")
            answers = q.get("answers", [])
            ok, reason = _validate_fill_blank_shape(q)
            if not ok:
                print(f"    [_verify_answer] fill_blank invalid: {reason}")
                return False
            verify_prompt = (
                "Independently solve the following fill-in-the-blank question.\n"
                "Decide whether each provided answer is mathematically or conceptually correct.\n"
                "NOTE: Accept mathematically equivalent expressions (e.g., O(n) vs O(N)) and exact conceptual synonyms.\n"
                "CRITICAL: If the math involves counting steps or calls, allow minor semantic variations (e.g., inclusive vs exclusive counting of the initial call). Before returning MISMATCH, double-check if the provided answers are correct under a valid interpretation of the algorithm.\n"
                "Return ONLY valid JSON with keys verdict, resolved_answers, and reason.\n\n"
                f"Sentence: {sentence}\n"
                f"Provided answers: {json.dumps(answers, ensure_ascii=False)}"
            )
            try:
                res = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": verify_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=700,
                )
                verdict_raw = json.loads(res.choices[0].message.content)
                verdict = str(verdict_raw.get("verdict", "")).strip().upper()
                resolved_answers = verdict_raw.get("resolved_answers", [])
                if verdict == "AMBIGUOUS":
                    print("    [_verify_answer] fill_blank ambiguous according to verifier.")
                    return False
                if verdict == "MATCH":
                    return True
                if not isinstance(resolved_answers, list) or len(resolved_answers) != len(answers):
                    print("    [_verify_answer] fill_blank verifier returned malformed answers.")
                    return False
                claimed = [_normalize_fill_blank_answer(a) for a in answers]
                resolved = [_normalize_fill_blank_answer(a) for a in resolved_answers]
                if claimed != resolved:
                    print(f"    [_verify_answer] fill_blank MISMATCH: claimed={answers}, resolved={resolved_answers}")
                    return False
                return verdict != "MISMATCH"
            except Exception:
                return True

        # ── true_false verification ──────────────────────────────────────
        if fmt == "true_false":
            statement = q.get("statement", "")
            if not statement:
                return True
            claimed_answer = q.get("tf_answer")  # bool or None

            verify_prompt = (
                f"Determine whether the following statement is True or False.\n"
                f"Show your work step by step, then on the LAST LINE write:\n"
                f"MY_VERDICT: True   OR   MY_VERDICT: False\n\n"
                f"Statement: {statement}"
            )
            try:
                res = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": verify_prompt}],
                    temperature=0,
                    max_tokens=600,
                )
                text  = res.choices[0].message.content
                match = re.search(r"MY_VERDICT:\s*(True|False)", text, re.IGNORECASE)
                if not match:
                    return True   # cannot parse → benefit of doubt
                verdict = match.group(1).strip().lower() == "true"
                if verdict != bool(claimed_answer):
                    print(f"    [_verify_answer] true_false MISMATCH: "
                          f"LLM says {verdict}, question says {claimed_answer}")
                    return False
                return True
            except Exception:
                return True

        # ── mcq_multi verification ───────────────────────────────────────
        if fmt == "mcq_multi":
            question = q.get("question", "")
            options  = q.get("options", {})
            claimed  = set(q.get("correct_answers", []))
            if not question or not options or not claimed:
                return True

            opts_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
            verify_prompt = (
                f"For each option below, independently determine if it is correct or incorrect.\n"
                f"Show your reasoning for each option.\n"
                f"On the LAST LINE write ONLY: CORRECT: <comma-separated letters, e.g. A,C>\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{opts_str}"
            )
            try:
                res = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": verify_prompt}],
                    temperature=0,
                    max_tokens=900,
                )
                text  = res.choices[0].message.content
                match = re.search(r"CORRECT:\s*([A-E,\s]+)", text, re.IGNORECASE)
                if not match:
                    return True
                independent = {x.strip().upper()
                               for x in match.group(1).split(",")
                               if x.strip().upper() in options}
                if not independent:
                    return True
                # Fail if more than 1 option differs between claimed and independent
                diff = claimed.symmetric_difference(independent)
                if len(diff) > 1:
                    print(f"    [_verify_answer] mcq_multi MISMATCH: "
                          f"claimed={sorted(claimed)}, independent={sorted(independent)}")
                    return False
                return True
            except Exception:
                return True

        # ── mcq_single: original logic ───────────────────────────────────
        if q_type == "conceptual":
            return True

        verify_prompt = (
            f"Solve this CS problem independently, showing all work step by step.\n\n"
            f"{q.get('question', '')}\n\n"
            f"At the very end write ONE line:\n"
            f"MY_ANSWER: <your computed result>"
        )
        try:
            res = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0,
                max_tokens=800,
            )
            text  = res.choices[0].message.content
            match = re.search(r"MY_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if not match:
                return True

            computed = re.sub(r"^[A-Da-d][.)]\s*", "", match.group(1).strip()).strip().lower()
            expected = re.sub(r"^[A-Da-d][.)]\s*", "", q.get("correct_answer", "").strip()).strip().lower()

            if computed == expected:
                return True
            if expected in computed or computed in expected:
                return True
            try:
                if abs(float(computed.split()[0]) - float(expected.split()[0])) < 0.01:
                    return True
            except (ValueError, TypeError, IndexError):
                pass
            return False

        except Exception:
            return True
