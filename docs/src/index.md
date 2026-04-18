# MemPalace.jl

**MemPalace.jl** is a Julia port of the
[MemPalace](https://github.com/MemPalace/mempalace) memory model — a
"memory palace" / method-of-loci inspired memory store organized around
**rooms**, **closets**, **drawers**, and **wings**. It complements the
other memory backends in the AgentFramework.jl ecosystem
(in-memory, Mem0, Graphiti) and integrates as both a drop-in
`AbstractMemoryStore` and a native `BaseContextProvider` via the
`MemPalaceAgentFrameworkExt` extension.

## Highlights

- `Palace{B,E}` — a parametric container holding any
  `AbstractPalaceBackend` for storage and any `AbstractMemPalaceEmbedder`
  for semantic similarity.
- Hybrid retrieval combining BM25 lexical scoring with cosine vector
  similarity, plus a "closet boost" that rewards results sharing the
  same closet (theme).
- `mine_text!` / `mine_conversation!` for ingesting raw text or chat
  history into structured drawers.
- Pluggable embedders: `DeterministicEmbedder` (CPU, deterministic) and
  `OllamaEmbedder` (live model).
- AgentFramework integration via `MemPalaceMemoryStore <: AbstractMemoryStore`
  (works with the standard `MemoryContextProvider`) and the native
  `MemPalaceContextProvider` with wing/room scoping from session metadata.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaKnowledge/MemPalace.jl")
```

See [Getting Started](guide/getting_started.md) for a walkthrough and
[API Reference](api.md) for the full surface.

## Module reference

```@docs
MemPalace
```
