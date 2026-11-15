"""
Data-structure tasks for the compositional-RL replication, written from the paper's description (He, Li, Wang and
Vitercik, arXiv 2609.19465, Appendix C). Their code is not released; DSR-Bench's prompt style is followed.

The idea in one picture. A *skill* is one operation on a data structure (insert, delete). An *atomic* task asks for
one skill on a given state; a *compound* task chains several, each step reading the state the last one wrote:

    atomic:    [3, 1, 4]  --insert(1, 9)-->  [3, 9, 1, 4]                       (one node)
    compound:  [3, 1, 4]  --insert(1, 9)-->  [3, 9, 1, 4]  --delete(0)-->  [9, 1, 4]  --insert(3, 2)-->  [9, 1, 4, 2]
                                                                               (a chain of nodes = a dependency graph)

Three levels (the paper's Figure 1):
    Level I   single-skill chain: insert, insert, insert ...           (horizon)
    Level II  multi-skill chain:  insert, delete, insert ...           (horizon + skill switching)
    Level III branch-merge graph: segment-tree construction            (+ non-local dependencies)

Every task has one right answer, computed here in code, so the RL reward is a plain equality check.
"""

from __future__ import annotations

import ast
import itertools
import json
import random
import re
from dataclasses import asdict, dataclass, field

DOMAINS = ["array", "heap", "bst", "hashmap", "bloom"]
SUFFIX = (
    "Approach the problem methodically. Ensure all conclusions are based on precise calculations and logical "
    "deductions. Feel free to explore various solution methods and cross-check results for consistency. Maintain "
    "dynamic thinking and always verify each step of your reasoning. The last line of your response should be of the "
    "following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$.' (without quotes), where ANSWER is just "
    "the final number or expression that solves the problem. Think carefully and break down the problem step by step."
)
P_INSERT = 0.7  # the paper's Level II mix: insert with probability 0.7, delete with 0.3
HASH_BUCKETS = 5
BLOOM_SIZE = 10


@dataclass
class Task:
    domain: str
    level: int  # 1, 2 or 3
    kind: str  # "atomic" or "compound"
    length: int
    prompt: str
    answer: object  # the ground-truth final state (JSON-able)
    ops: list = field(default_factory=list)  # the operations, in order
    states: list = field(default_factory=list)  # ground-truth state after each op (states[0] = initial)
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def to_json(self):
        return asdict(self)


# ---------------------------------------------------------------- the five data structures (skills as functions)


def array_apply(state, op):
    s = list(state)
    if op[0] == "insert":
        s.insert(op[1], op[2])
    else:
        del s[op[1]]
    return s


def heap_apply(state, op):
    """Array-based min-heap. Insert: append, sift up. Delete: remove the root, move the last element to the root,
    sift down; on equal children go left; swap only if the child is strictly smaller."""
    h = list(state)
    if op[0] == "insert":
        h.append(op[1])
        i = len(h) - 1
        while i > 0:
            p = (i - 1) // 2
            if h[i] < h[p]:
                h[i], h[p] = h[p], h[i]
                i = p
            else:
                break
    else:
        last = h.pop()
        if h:
            h[0] = last
            i = 0
            while True:
                left, right = 2 * i + 1, 2 * i + 2
                if left >= len(h):
                    break
                child = left
                if right < len(h) and h[right] < h[left]:
                    child = right
                if h[child] < h[i]:
                    h[i], h[child] = h[child], h[i]
                    i = child
                else:
                    break
    return h


def bst_insert(tree, v):
    """tree: dict value -> [left, right]; the root is stored as tree[None] = [root]."""
    t = {k: list(c) for k, c in tree.items()}
    root = t.get(None, [None])[0]
    if root is None:
        t[None] = [v]
        t[v] = [None, None]
        return t
    cur = root
    while True:
        side = 0 if v < cur else 1
        nxt = t[cur][side]
        if nxt is None:
            t[cur][side] = v
            t[v] = [None, None]
            return t
        cur = nxt


def bst_remove(tree, v):
    """Standard deletion: a leaf is removed; a node with one child is replaced by it; a node with two children takes
    the value of its in-order successor (smallest in the right subtree), and that successor node is removed."""

    class N:
        __slots__ = ("l", "r", "v")

        def __init__(self, v):
            self.v, self.l, self.r = v, None, None

    def build(t, x):
        if x is None:
            return None
        n = N(x)
        n.l = build(t, t[x][0])
        n.r = build(t, t[x][1])
        return n

    def delete(n, x):
        if n is None:
            raise KeyError(x)
        if x < n.v:
            n.l = delete(n.l, x)
            return n
        if x > n.v:
            n.r = delete(n.r, x)
            return n
        if n.l is None:
            return n.r
        if n.r is None:
            return n.l
        s = n.r
        while s.l is not None:
            s = s.l
        n.v = s.v
        n.r = delete(n.r, s.v)
        return n

    root = build(tree, tree.get(None, [None])[0])
    root = delete(root, v)
    out = {None: [root.v if root else None]}

    def flat(n):
        if n is None:
            return
        out[n.v] = [n.l.v if n.l else None, n.r.v if n.r else None]
        flat(n.l)
        flat(n.r)

    flat(root)
    return out


def bst_preorder(t):
    out = []

    def go(x):
        if x is None:
            return
        out.append(x)
        go(t[x][0])
        go(t[x][1])

    go(t.get(None, [None])[0])
    return out


def bst_from_preorder(pre):
    """A BST is fully determined by its pre-order list: inserting the values in that order rebuilds it."""
    t = {None: [None]}
    for v in pre:
        t = bst_insert(t, v)
    return t


def bst_describe(t):
    root = t.get(None, [None])[0]
    if root is None:
        return "The tree is empty."
    lines = [f"Root: {root}"]
    for v in bst_preorder(t):
        left, right = t[v]
        lines.append(
            f"Node {v}: left child = {left if left is not None else 'None'}, "
            f"right child = {right if right is not None else 'None'}"
        )
    return "\n".join(lines)


def bst_apply(pre, op):
    t = bst_from_preorder(pre)
    t = bst_insert(t, op[1]) if op[0] == "insert" else bst_remove(t, op[1])
    return bst_preorder(t)


def hashmap_apply(buckets, op):
    b = [[list(kv) for kv in bucket] for bucket in buckets]
    k = op[1]
    idx = k % len(b)
    if op[0] == "insert":
        for kv in b[idx]:
            if kv[0] == k:
                kv[1] = op[2]
                break
        else:
            b[idx].append([k, op[2]])
    else:
        b[idx] = [kv for kv in b[idx] if kv[0] != k]
    return b


def bloom_positions(x):
    return [x % BLOOM_SIZE, (3 * x + 1) % BLOOM_SIZE]


def bloom_apply(counts, op):
    c = list(counts)
    for p in bloom_positions(op[1]):
        if op[0] == "insert":
            c[p] += 1
        else:
            c[p] = max(0, c[p] - 1)
    return c


APPLY = {"array": array_apply, "heap": heap_apply, "bst": bst_apply, "hashmap": hashmap_apply, "bloom": bloom_apply}

# ---------------------------------------------------------------- prompts

RULES = {
    "array": (
        "An array supports the following operations (using 0-based indexing):\n"
        "1. (insert, index=i, value=v) inserts value v at position i, shifting elements at i and beyond to the right.\n"
        "2. (delete, index=i) removes the element currently at position i."
    ),
    "heap": (
        "A min-heap is stored as an array: the children of position i are positions 2i+1 and 2i+2, and every parent "
        "is less than or equal to its children. It supports:\n"
        "1. (insert, value=v): append v at the end, then sift up: while v is strictly smaller than its parent, swap "
        "them.\n"
        "2. (delete): remove the root. Move the last element to the root, then sift down: while it has a child "
        "strictly smaller than it, swap it with its smaller child (if both children are equal, use the left one)."
    ),
    "bst": (
        "A binary search tree (BST) keeps smaller values in the left subtree and larger values in the right subtree. "
        "All values are distinct. It supports:\n"
        "1. (insert, value=v): walk down from the root, going left if v is smaller and right if larger, and attach v "
        "as a new leaf.\n"
        "2. (remove, value=v): if the node is a leaf, delete it; if it has one child, replace it by that child; if it "
        "has two children, replace its value with its in-order successor (the smallest value in its right subtree) "
        "and remove that successor node from the right subtree."
    ),
    "hashmap": (
        f"A hashmap has {HASH_BUCKETS} buckets numbered 0 to {HASH_BUCKETS - 1}. The bucket of key k is "
        f"k mod {HASH_BUCKETS}. Each bucket is a list of [key, value] pairs. It supports:\n"
        "1. (insert, key=k, value=v): if k is already in its bucket, update its value to v; otherwise append [k, v] "
        "to the end of the bucket.\n"
        "2. (delete, key=k): remove the pair with key k from its bucket (if present)."
    ),
    "bloom": (
        f"A counting Bloom filter has an array of {BLOOM_SIZE} counters (positions 0 to {BLOOM_SIZE - 1}) and two "
        f"hash functions: h1(x) = x mod {BLOOM_SIZE} and h2(x) = (3x + 1) mod {BLOOM_SIZE}. It supports:\n"
        "1. (insert, value=x): add 1 to the counter at h1(x) and add 1 to the counter at h2(x) (if both are the same "
        "position, that counter goes up by 2).\n"
        "2. (delete, value=x): subtract 1 from the counter at h1(x) and from the counter at h2(x), never going below 0."
    ),
}

ANSWER_FORMAT = {
    "array": "Your answer should be a list of integers, e.g. [1, 2, 3].",
    "heap": "Your answer should be the heap array as a list of integers, e.g. [1, 2, 3].",
    "bst": (
        "Your answer should be the pre-order traversal (root, left subtree, right subtree) of the final tree as a "
        "list of integers, e.g. [5, 3, 8]."
    ),
    "hashmap": (
        "Your answer should be the list of all buckets in order, each a list of [key, value] pairs, e.g. "
        "[[[5, 1]], [], [[2, 7], [12, 3]], [], []]."
    ),
    "bloom": "Your answer should be the list of counters, e.g. [0, 1, 0, 2, 0, 0, 1, 0, 0, 0].",
}


def fmt_op(domain, op):
    if domain == "array":
        return f"(insert, index={op[1]}, value={op[2]})" if op[0] == "insert" else f"(delete, index={op[1]})"
    if domain == "heap":
        return f"(insert, value={op[1]})" if op[0] == "insert" else "(delete)"
    if domain == "bst":
        return f"(insert, value={op[1]})" if op[0] == "insert" else f"(remove, value={op[1]})"
    if domain == "hashmap":
        return f"(insert, key={op[1]}, value={op[2]})" if op[0] == "insert" else f"(delete, key={op[1]})"
    return f"(insert, value={op[1]})" if op[0] == "insert" else f"(delete, value={op[1]})"


def describe_state(domain, state):
    if domain == "array":
        return f"Initial array: {state}"
    if domain == "heap":
        return f"Initial heap array: {state}"
    if domain == "bst":
        return "Initial tree:\n" + bst_describe(bst_from_preorder(state))
    if domain == "hashmap":
        return f"Initial buckets: {json.dumps(state)}"
    return f"Initial counters: {state}"


def make_prompt(domain, state, ops):
    many = len(ops) > 1
    if many:
        head = "Start with the provided initial state and apply each operation in order."
        q = "What is the final state after applying all operations?"
    else:
        head = "Start with the provided initial state and apply the operation."
        q = "What is the state after applying the operation?"
    op_lines = "\n".join(fmt_op(domain, op) for op in ops)
    return (
        f"{RULES[domain]}\n{head}\n{describe_state(domain, state)}\nOperation{'s' if many else ''}:\n{op_lines}\n"
        f"Q: {q}\n{ANSWER_FORMAT[domain]}\n{SUFFIX}"
    )


# ---------------------------------------------------------------- random states and operations


def initial_state(domain, n, rng):
    """Returns (state, stored); `stored` lists the Bloom filter's inserted values (None for other domains)."""
    if domain == "array":
        return [rng.randint(0, 99) for _ in range(n)], None
    if domain == "heap":
        h = []
        for _ in range(n):
            h = heap_apply(h, ("insert", rng.randint(0, 99)))
        return h, None
    if domain == "bst":
        return bst_preorder(bst_from_preorder(rng.sample(range(100), n))), None
    if domain == "hashmap":
        b = [[] for _ in range(HASH_BUCKETS)]
        for k in rng.sample(range(100), n):
            b = hashmap_apply(b, ("insert", k, rng.randint(0, 99)))
        return b, None
    c, stored = [0] * BLOOM_SIZE, []
    for _ in range(n):
        x = rng.randint(0, 99)
        c = bloom_apply(c, ("insert", x))
        stored.append(x)
    return c, stored


def random_op(domain, state, rng, allow_delete, stored=None):
    """One random valid operation. Deletes are always valid (an existing index, key or value), as in the paper."""
    if domain == "bloom":
        size = len(stored or [])
    elif domain == "hashmap":
        size = sum(len(b) for b in state)
    else:
        size = len(state)
    delete = allow_delete and size > 0 and rng.random() >= P_INSERT
    if domain == "array":
        if delete:
            return ("delete", rng.randrange(len(state)))
        return ("insert", rng.randint(0, len(state)), rng.randint(0, 99))
    if domain == "heap":
        return ("delete",) if delete else ("insert", rng.randint(0, 99))
    if domain == "bst":
        if delete:
            return ("remove", rng.choice(state))
        return ("insert", rng.choice(sorted(set(range(100)) - set(state))))
    if domain == "hashmap":
        keys = [kv[0] for b in state for kv in b]
        if delete:
            return ("delete", rng.choice(keys))
        k = rng.choice(keys) if keys and rng.random() < 0.2 else rng.randint(0, 99)  # sometimes an update
        return ("insert", k, rng.randint(0, 99))
    if delete:
        return ("delete", rng.choice(stored))
    return ("insert", rng.randint(0, 99))


def make_chain_task(domain, level, kind, length, seed):
    """Level I (insert only) or II (insert + delete). Length = initial size; a compound task also has `length` ops."""
    rng = random.Random(f"{domain}-{level}-{kind}-{length}-{seed}")
    init, stored = initial_state(domain, length, rng)
    n_ops = length if kind == "compound" else 1
    states, ops, s = [init], [], init
    for _ in range(n_ops):
        op = random_op(domain, s, rng, allow_delete=(level == 2), stored=stored)
        if domain == "bloom":
            if op[0] == "insert":
                stored.append(op[1])
            else:
                stored.remove(op[1])
        s = APPLY[domain](s, op)
        ops.append(list(op))
        states.append(s)
    kinds = [o[0] for o in ops]
    switches = sum(1 for a, b in itertools.pairwise(kinds) if a != b)
    prompt = make_prompt(domain, init, ops)
    return Task(domain, level, kind, length, prompt, states[-1], ops, states, seed, {"switches": switches})


# ---------------------------------------------------------------- Level III: segment-tree construction

SEG_RULES = (
    "A segment tree over an integer array x[0..n-1] is a binary tree. Every node covers an inclusive index interval "
    "[l, r] and stores a summary [sum, best_prefix, best_suffix, best_subarray]: the total of x[l..r], the largest sum "
    "of a non-empty prefix of it, the largest sum of a non-empty suffix of it, and the largest sum of any non-empty "
    "contiguous piece of it. It is built recursively from the root interval [0, n-1]:\n"
    "- SplitInterval: an interval [l, r] with l < r splits at m = floor((l + r) / 2) into children [l, m] and "
    "[m+1, r].\n"
    "- MakeLeaf: an interval [i, i] is a leaf with summary [x[i], x[i], x[i], x[i]].\n"
    "- MergeSummaries: a parent with left summary [Ls, Lp, Lx, Lb] and right summary [Rs, Rp, Rx, Rb] has summary "
    "[Ls + Rs, max(Lp, Ls + Rp), max(Rx, Rs + Lx), max(Lb, Rb, Lx + Rp)]."
)
SEG_FORMAT = (
    "Your answer should be the pre-order traversal (node, left subtree, right subtree) of the whole tree as a list of "
    "nodes, each written [[l, r], [sum, best_prefix, best_suffix, best_subarray]], e.g. for x = [2, -1]: "
    "[[[0, 1], [1, 2, 1, 2]], [[0, 0], [2, 2, 2, 2]], [[1, 1], [-1, -1, -1, -1]]]."
)


def seg_merge(L, R):
    return [L[0] + R[0], max(L[1], L[0] + R[1]), max(R[2], R[0] + L[2]), max(L[3], R[3], L[2] + R[1])]


def seg_build(x, l, r, out, order):
    """Pre-order node list in `out`; the skill invocations in execution order (one topological order) in `order`."""
    node = [[l, r], None]
    out.append(node)
    if l == r:
        order.append(("MakeLeaf", [l, x[l]]))
        node[1] = [x[l]] * 4
        return node[1]
    m = (l + r) // 2
    order.append(("SplitInterval", [l, r]))
    left = seg_build(x, l, m, out, order)
    right = seg_build(x, m + 1, r, out, order)
    node[1] = seg_merge(left, right)
    order.append(("MergeSummaries", [left, right]))
    return node[1]


def segment_answer(x):
    out, order = [], []
    seg_build(x, 0, len(x) - 1, out, order)
    return out, order


def make_segment_task(kind, length, seed):
    rng = random.Random(f"seg-{kind}-{length}-{seed}")
    x = [rng.randint(-20, 20) for _ in range(length)]
    ans, order = segment_answer(x)
    if kind == "compound":
        prompt = (
            f"{SEG_RULES}\nArray: x = {x}\nQ: Build the segment tree over [0, {length - 1}].\n{SEG_FORMAT}\n{SUFFIX}"
        )
        return Task("segment", 3, kind, length, prompt, ans, [list(o) for o in order], [], seed, {"n_nodes": len(ans)})
    # atomic: one skill invocation drawn from a real construction over an array of this length
    skill, args = rng.choice(order)
    if skill == "SplitInterval":
        l, r = args
        m = (l + r) // 2
        q = f"Apply SplitInterval to the interval [{l}, {r}]."
        fmt = "Your answer should be the two child intervals as a list, e.g. [[0, 1], [2, 3]]."
        answer = [[l, m], [m + 1, r]]
    elif skill == "MakeLeaf":
        i, v = args
        q = f"Array: x = {x}\nApply MakeLeaf to the interval [{i}, {i}]."
        fmt = "Your answer should be the leaf summary as a list of four integers, e.g. [3, 3, 3, 3]."
        answer = [v] * 4
    else:
        L, R = args
        q = f"Apply MergeSummaries to the left summary {L} and the right summary {R}."
        fmt = "Your answer should be the parent summary as a list of four integers, e.g. [1, 2, 1, 2]."
        answer = seg_merge(L, R)
    prompt = f"{SEG_RULES}\nQ: {q}\n{fmt}\n{SUFFIX}"
    return Task("segment", 3, "atomic", length, prompt, answer, [[skill, args]], [], seed, {"skill": skill})


def make_task(domain, level, kind, length, seed):
    if level == 3:
        return make_segment_task(kind, length, seed)
    return make_chain_task(domain, level, kind, length, seed)


def atomic_from_state(domain, state, op):
    """An atomic prompt for one op on a given state (used by the step-by-step scaffold and the per-step test)."""
    return make_prompt(domain, state, [op])


# ---------------------------------------------------------------- answer extraction and checking


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = i + len("\\boxed")
    while j < len(text) and text[j] == " ":
        j += 1
    if j < len(text) and text[j] == "{":
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    return text[j + 1 : k]
        return text[j + 1 :]
    m = re.match(r"\s*(\S+)", text[j:])
    return m.group(1) if m else None


def first_balanced(s):
    """The first balanced [...] in s."""
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    for k in range(start, len(s)):
        if s[k] == "[":
            depth += 1
        elif s[k] == "]":
            depth -= 1
            if depth == 0:
                return s[start : k + 1]
    return None


def to_lists(x):
    if isinstance(x, (list, tuple)):
        return [to_lists(v) for v in x]
    return x


def parse_answer(text):
    """Model reply -> a Python structure, or None. The last \\boxed{}, else what follows "final answer is"."""
    cand = last_boxed(text)
    if cand is None:
        m = re.search(r"final answer is[:\s]*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        cand = m.group(1) if m else None
    if cand is None:
        return None
    for a, b in [("\\text", ""), ("\\left", ""), ("\\right", ""), ("$", ""), ("\\[", "["), ("\\]", "]")]:
        cand = cand.replace(a, b)
    cand = cand.replace("null", "None").strip().rstrip(".").strip()
    # The whole answer as a literal first: a bare tuple such as \boxed{39, 77, 56} or \boxed{[a], [b]} reads as a
    # tuple, and tuples and lists count as the same (as in the paper).
    try:
        val = ast.literal_eval(cand)
        if isinstance(val, (list, tuple)):
            return to_lists(val)
    except Exception:
        pass
    lst = first_balanced(cand)
    if lst is None:
        return None
    try:
        return to_lists(ast.literal_eval(lst))
    except Exception:
        try:
            return to_lists(json.loads(lst))
        except Exception:
            return None


def canonical(domain, x):
    """Hashmap pairs may appear in any order inside a bucket (as in the paper); everything else must match exactly."""
    if x is None:
        return None
    if domain == "hashmap":
        try:
            return [sorted(list(kv) for kv in b) for b in x]
        except Exception:
            return None
    return x


def is_correct(domain, text, answer):
    return canonical(domain, parse_answer(text)) == canonical(domain, answer)


def reward(task: dict, text: str) -> float:
    return 1.0 if is_correct(task["domain"], text, task["answer"]) else 0.0
