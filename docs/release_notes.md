# Source layout

The implementation lives in `src/keyrec/`, with Gemma and NEO-ov wrappers in
`models/`. Evaluation includes four benchmark runners, default profiles, command
planning.

Model weights, datasets, and generated run outputs are stored separately from
the source checkout. Third-party code and its original license notices are
listed in `THIRD_PARTY_NOTICES.md`.
