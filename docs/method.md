# Memory method and default profiles

KeyRec operates on model-facing visual embeddings. Memory writing does not use
the question. Reading selects visual evidence from the stored memory without
re-encoding historical frames.

## Memory construction

The recent cache appends each frame's tokens and retains the newest `N_C` tokens.
For each frame, token novelty is one minus the maximum cosine similarity to a
stored event route. The `m` most novel tokens form a candidate payload; a
softmax-weighted sum forms its normalized route, and mean novelty is its utility.

The bounded event bank applies an add–merge–evict update:

- Merge with the most similar event when route similarity is at least `0.75`
  and its forward temporal gap is at most `2.0` seconds.
- Update the route with normalized moving-average weight `0.2`.
- Time-sort tokens and their metadata together, then uniformly downsample the
  merged payload to `m` tokens.
- Retain the maximum candidate utility seen by an event.
- On overflow, evict the event with the lowest utility plus weighted distance
  to its nearest temporal neighbor.

## Default profiles

| Setting | Frames | Nominal tokens/frame | `m` | Event capacity | Recent/readout budget | Coverage weight |
|---|---:|---:|---:|---:|---:|---:|
| Gemma streaming | 32 | 70 | 16 | 16 | 224 | 0.2 |
| NEO-ov streaming | 32 | 210 | 48 | 16 | 672 | 0.2 |
| Gemma long video | 64 | 70 | 16 | 32 | 448 | 0.0 |
| NEO-ov long video | 64 | 210 | 48 | 32 | 1344 | 0.0 |

OVO-Bench and StreamingBench use the streaming profiles. LongVideoBench and
Video-MME-v2 use the long-video profiles. These settings are loaded from
`configs/main_results.yaml`.

## Query-adaptive readout

The frozen model routes the question with a text-only pass. The prompt includes
only the question and a description of the two memory sources. The exact prompts
are defined by `build_router_prompt` in `src/keyrec/readout.py`.

| State | Streaming events | Long-video events | Retrieval |
|---|---:|---:|---|
| `all_recent` | 0 | 0 | Recent tail |
| `heavy_recent` | 2 | 8 | Query relevance |
| `balanced` | 4 | 14 | Query relevance |
| `heavy_key_event` | 12 | 24 | Temporal coverage |
| `all_key_event` | 14 | 28 | Temporal coverage |

Each selected event contributes its complete payload. Events are restored to
chronological order; the newest recent tokens fill the remaining readout budget.
Insufficient stored evidence leaves part of the budget unused. The serialized
readout uses neutral numbered segment labels and token counts.

## Model adapters

Gemma extracts projected vision tokens, then injects selected embeddings through
its video interface. Its vision encoder still processes sampled frames.

NEO-ov retains each token's source time and original height/width indexes.
Readout groups tokens into chronological source-frame spans, restores spatial
indexes, and inserts embeddings at the corresponding `<IMG_CONTEXT>` positions.
