# Split-K reduction: derivation

Written before `src/kernel_v4_split_k.py`, as a correctness spec to
implement against rather than inventing the reduction logic ad hoc while
debugging.

## Why split-K

v1/v2/v3 grid over `(batch, num_kv_heads)`. At batch=1, `num_kv_heads=2`
(the realistic single-request decode scenario), that's 2 thread blocks on
a ~28-SM GPU — confirmed via NCU (`profiles/notes.md`, v1 section: 8.33%
occupancy, `long_scoreboard`-dominant stall). Split-K adds a third grid
dimension over chunks of the sequence, so a single (batch, kv_head) pair
can occupy many blocks instead of one.

## Phase 1: per-chunk partial softmax statistics

For split `s` covering token range `[start_s, end_s)`, phase 1 runs the
same online-softmax loop v1/v2/v3 already use, just bounded to that
range instead of `[0, seq_len)`, and stops one step short of normalizing:

```
m_s = max(scores for tokens in [start_s, end_s))
l_s = sum(exp(scores - m_s))
O_s = sum(exp(scores - m_s) * V)          # NOT divided by l_s
```

`(O_s, m_s, l_s)` is stored per `(batch, kv_head, split, gqa_row)`.

**Empty chunks.** If `start_s >= seq_len` (more splits requested than the
sequence has tokens for), the chunk is empty: `m_s = -inf`, `l_s = 0`,
`O_s = 0`. Split 0 always has `start_0 = 0 < seq_len`, since `seq_len >=
1` is already this project's documented precondition (v1's docstring) —
so split 0 is always non-empty, though the phase-2 merge below does not
rely on that ordering for correctness (see the guard below).

## Phase 2: merging partials

Standard online-softmax merge, applied across splits instead of across
token tiles — same algebra v1's loop already uses internally, just
combining precomputed `(O_s, m_s, l_s)` triples instead of computing them
from raw K/V:

```
m = max_s(m_s)
l = sum_s(l_s * exp(m_s - m))
O = sum_s(O_s * exp(m_s - m)) / l
```

Implemented as a running fold over splits (`tl.static_range(NUM_SPLITS)`,
a compile-time-unrolled loop — confirmed exempt from `tl.arange`'s
power-of-2 constraint by reading Triton's compiler source directly, not
assumed):

```
m_run, l_run, acc_run = -inf, 0, 0
for s in static_range(NUM_SPLITS):
    m_new  = max(m_run, m_s)
    alpha  = exp(m_run - m_new)     # rescale the running accumulator
    beta_s = exp(m_s   - m_new)     # rescale split s's contribution
    l_run   = l_run * alpha + l_s * beta_s
    acc_run = acc_run * alpha + O_s * beta_s
    m_run = m_new
final: O = acc_run / l_run
```

## The `-inf - -inf` guard

`beta_s = exp(m_s - m_new)` is only safe if `m_s` is finite whenever it's
used in a subtraction against another `-inf`. With split 0 always
non-empty and `static_range` iterating splits in order 0..N-1, `m_run`
becomes finite on the very first fold — but making correctness *depend*
on that ordering, rather than being robust regardless of it, is fragile:
a future change to iteration order, or relaxing the `seq_len >= 1`
precondition, would silently reintroduce `exp(-inf - -inf) = exp(nan) =
nan`.

Guard explicitly instead of relying on ordering:

```
beta_s = tl.where(m_s == float("-inf"), 0.0, tl.exp(m_s - m_new))
```

An empty split contributes exactly `0` to both `l_run` and `acc_run`
regardless of where it falls in the fold — correct by construction, not
by accident of iteration order.

## Sanity anchor: `num_splits=1` must equal v1

At `num_splits=1`, phase 1's loop covers the entire sequence in one
"split" — byte-for-byte the same computation as v1's loop — and phase 2's
merge degenerates to a single term: `m = m_0`, `l = l_0`, `O = O_0 / l_0`,
exactly v1's final `acc / l_i`. `paged_attention_decode_v4(..., num_splits=1)`
should therefore match `paged_attention_decode_v1(...)` at the same
seed/shape to a *tight* tolerance (not the fp16-vs-fp32 tolerance used
against the reference oracle) — this is a much stronger check than "close
to the reference," since it confirms split-K is a pure reassociation of
v1's math rather than a different algorithm that happens to also pass a
loose tolerance. See `tests/test_kernel_v4.py`.
