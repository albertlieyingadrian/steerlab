"""The chain-following task and the model shapes compared in the looped-transformer note.

Task ("which digit does f hold?"). An example is a shuffled set of N_STATEMENTS assignments over distinct variable
names, such as `c=b`, `b=a`, `a=7`, plus a query variable. The assignments form chains that each end in a digit. The
answer is the digit at the end of the query variable's chain; the number of assignments between the query and that
digit is the number of hops. Every example has exactly N_STATEMENTS assignments, so a longer query chain means fewer
distractor assignments, never a longer input: chain length is the only thing the hop count changes.

Each assignment is one input position, embedded as lhs_embedding[a] + rhs_embedding[b]. There are no position
embeddings (the order is shuffled and carries nothing) and no causal mask. The query is one more position. The model
reads the answer off the query position.

Models (all d_model 128, 4 heads, pre-norm blocks):
- `standard<L>`: L distinct blocks, run once.
- `looped`: a prelude of P blocks, a recurrent core of R blocks with shared weights run k times, then a coda of C
  blocks. Each loop re-adds the prelude's output e, as Dory does: s_i = core(s_{i-1} + e), s_0 = e. k is an argument
  of forward(), so one checkpoint can run at any k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

N_VARS = 96  # variable names
N_DIGITS = 10
N_STATEMENTS = 48  # so a 24-hop query still leaves 24 distractor assignments
# rhs vocabulary: 0..N_VARS-1 are variables, N_VARS..N_VARS+9 are digits
RHS_VOCAB = N_VARS + N_DIGITS
TRAIN_HOPS = (1, 8)  # hop counts seen in training (inclusive)
MAX_HOPS = 24
D_MODEL, N_HEADS = 128, 4


@dataclass
class Batch:
    lhs: torch.Tensor  # (B, N) variable ids
    rhs: torch.Tensor  # (B, N) rhs ids (variable or N_VARS + digit)
    query: torch.Tensor  # (B,) variable id
    answer: torch.Tensor  # (B,) digit 0-9
    hops: torch.Tensor  # (B,)
    chain: torch.Tensor  # (B, MAX_HOPS + 1) chain[j] = variable j hops on from the query (chain[0] = query); -1 padded
    chain_pos: torch.Tensor  # (B, MAX_HOPS + 1) input position of the assignment whose lhs is chain[j]; -1 padded

    def to(self, dev):
        return Batch(*(getattr(self, f).to(dev) for f in self.__dataclass_fields__))


def make_batch(
    rng: np.random.Generator, batch: int, hops_range=TRAIN_HOPS, hops: int | None = None, twin: bool = False
) -> Batch:
    """Random examples. Query chain length uniform in hops_range (or fixed `hops`); the other assignments are split
    into distractor chains of random length 1-8 (the last one cut to fit), each ending in its own random digit.
    twin=True (the control set, added after the results): the first distractor chain has exactly the query's length,
    so chain length cannot pick out the query's chain (needs 2 x hops <= 48).
    Vectorised over the batch: positions are first laid out chain by chain (query chain first), then shuffled."""
    B, N = batch, N_STATEMENTS
    h = np.full(B, hops) if hops is not None else rng.integers(hops_range[0], hops_range[1] + 1, B)
    lead = 2 * h if twin else h  # where the random 1-8 distractors start
    names = np.argsort(rng.random((B, N_VARS)), 1)[:, :N]  # distinct names per example
    # chain starts: 0, h, then h + cumulative sums of lengths drawn uniformly from 1-8
    starts = np.zeros((B, N), bool)
    starts[:, 0] = True
    starts[np.arange(B), h % N] = True  # h < N always
    if twin:
        starts[np.arange(B)[2 * h < N], (2 * h)[2 * h < N]] = True  # the twin chain occupies positions h .. 2h-1
    ends = lead[:, None] + np.cumsum(rng.integers(1, 9, (B, N)), 1)
    bb, cc = np.nonzero(ends < N)
    starts[bb, ends[bb, cc]] = True
    chain_id = np.cumsum(starts, 1) - 1
    digit_of_chain = rng.integers(0, N_DIGITS, (B, N))
    digit = np.take_along_axis(digit_of_chain, chain_id, 1)
    nxt_is_same = np.zeros((B, N), bool)
    nxt_is_same[:, :-1] = ~starts[:, 1:]
    nxt_name = np.zeros((B, N), np.int64)
    nxt_name[:, :-1] = names[:, 1:]
    lhs_sorted = names
    rhs_sorted = np.where(nxt_is_same, nxt_name, N_VARS + digit)
    order = np.argsort(rng.random((B, N)), 1)  # shuffled position p holds sorted assignment order[p]
    inv = np.argsort(order, 1)
    lhs = np.take_along_axis(lhs_sorted, order, 1)
    rhs = np.take_along_axis(rhs_sorted, order, 1)
    j = np.arange(MAX_HOPS + 1)[None, :]
    in_chain = j < h[:, None]
    jj = np.minimum(j, N - 1).repeat(B, 0)
    chain = np.where(in_chain, np.take_along_axis(names, jj, 1), -1)
    chain_pos = np.where(in_chain, np.take_along_axis(inv, jj, 1), -1)
    query, answer = names[:, 0], digit_of_chain[:, 0]
    return Batch(
        *(
            torch.from_numpy(np.ascontiguousarray(x).astype(np.int64))
            for x in (lhs, rhs, query, answer, h, chain, chain_pos)
        )
    )


class Block(nn.Module):
    def __init__(self, d=D_MODEL, h=N_HEADS):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class Embed(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.lhs = nn.Embedding(N_VARS, d)
        self.rhs = nn.Embedding(RHS_VOCAB, d)
        self.q = nn.Embedding(N_VARS, d)  # the query has its own table: "what does v hold?"

    def forward(self, batch: Batch):
        x = self.lhs(batch.lhs) + self.rhs(batch.rhs)
        return torch.cat([x, self.q(batch.query)[:, None]], 1)  # query is the last position


class Standard(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.embed = Embed()
        self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_DIGITS)
        self.n_layers = n_layers

    def forward(self, batch: Batch, k=None, return_states=False):
        x = self.embed(batch)
        states = [x]
        for blk in self.blocks:
            x = blk(x)
            states.append(x)
        logits = self.head(self.norm(x[:, -1]))
        return (logits, states) if return_states else logits

    def depth(self, k=None):
        return self.n_layers


class Looped(nn.Module):
    """inject: 'add' (default, as in Dory: the prelude output e is re-added to the state before every loop, s + e, no
    extra parameters); 'concat' (Geiping et al.'s form: adapter([norm(s); e]), a new linear map on every loop's input;
    the pilot's version, which stalled); 'none' (the state alone, no re-injection)."""

    def __init__(self, prelude=1, core=2, coda=1, inject="add"):
        super().__init__()
        self.embed = Embed()
        self.prelude = nn.ModuleList(Block() for _ in range(prelude))
        self.core = nn.ModuleList(Block() for _ in range(core))
        self.coda = nn.ModuleList(Block() for _ in range(coda))
        self.inject = inject
        if inject == "concat":
            self.state_norm = nn.LayerNorm(D_MODEL)
            self.adapter = nn.Linear(2 * D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_DIGITS)
        self.sizes = (prelude, core, coda)

    def run_core(self, s, e):
        if self.inject == "add":
            x = s + e
        elif self.inject == "concat":
            x = self.adapter(torch.cat([self.state_norm(s), e], -1))
        else:
            x = s
        for blk in self.core:
            x = blk(x)
        return x

    def readout(self, s):
        for blk in self.coda:
            s = blk(s)
        return self.head(self.norm(s[:, -1]))

    def forward(self, batch: Batch, k=4, return_states=False, readout_every=False):
        """k loops. return_states: the state after the prelude and after every loop. readout_every: the coda's logits
        after every loop (a list of k tensors), i.e. the answer if the model stopped after loop i."""
        x = self.embed(batch)
        for blk in self.prelude:
            x = blk(x)
        e = s = x
        states, outs = [e], []
        for _ in range(k):
            s = self.run_core(s, e)
            states.append(s)
            if readout_every:
                outs.append(self.readout(s))
        if readout_every:
            return outs, states
        logits = self.readout(s)
        return (logits, states) if return_states else logits

    def depth(self, k=4):
        p, c, o = self.sizes
        return p + c * k + o


def build(spec: str) -> nn.Module:
    """'standard4', 'standard10', 'looped' (random k), 'looped_fixed4' (same shape, k = 4 always), 'looped_noinject'
    (random k, no re-injection), 'looped_concat' (random k, concatenate-and-project injection)."""
    if spec.startswith("standard"):
        return Standard(int(spec.removeprefix("standard")))
    if spec == "looped_noinject":
        return Looped(inject="none")
    if spec == "looped_concat":
        return Looped(inject="concat")
    return Looped()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shortcut_baseline(batch: Batch) -> float:
    """Expected accuracy of a model that never follows the chain: it picks uniformly among the chain-end digits
    present in the input (one per chain). Averaged over the batch."""
    p = 0.0
    rhs, ans = batch.rhs.numpy(), batch.answer.numpy()
    for b in range(len(ans)):
        ends = rhs[b][rhs[b] >= N_VARS] - N_VARS
        p += float((ends == ans[b]).mean())
    return p / len(ans)
