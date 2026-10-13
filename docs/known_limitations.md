# Known limitations

- Memory updates use fixed novelty, merge, and retention rules.
- Gemma memory selection happens after vision encoding and does not reduce
  vision-encoder work.
- A bounded readout may omit fine visual details or evidence spread across time.
- Events retain visual patches and timestamps without explicitly tracking
  objects, trajectories, durations, or state transitions.
- The text-only router adds a model pass and uses a discrete allocation codebook.
- Each backbone requires a compatible visual-token injection path. NEO-ov
  additionally requires preservation of temporal and spatial token indexes.
