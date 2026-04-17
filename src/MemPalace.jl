"""
    MemPalace

A Julia port of [MemPalace](https://github.com/MemPalace/mempalace) — a
local-first verbatim AI memory system organised as a *palace* of *wings*
(entity scopes), *rooms* (topic scopes), *drawers* (verbatim text chunks),
and *closets* (compact topic-pointer indexes).

## Modules

  * `MemPalace.Drawer` — verbatim text chunk with wing/room metadata.
  * `MemPalace.Closet` — packed topic pointer line(s) into one or more
    drawers.
  * `MemPalace.Palace` — high-level orchestrator over a backend +
    embedder. Use `mine_text!` to ingest, `search` / `wake_up` to query.

## Backends

The retrieval layer is pluggable via [`AbstractPalaceBackend`](@ref). The
default is [`InMemoryPalaceBackend`](@ref), which keeps drawers and
closets in memory with a brute-force cosine + BM25 hybrid search. Drop-in
backends can be supplied without touching the rest of the pipeline,
mirroring the upstream Python design (see `mempalace/backends/base.py`).

## Embedders

Embeddings come from an [`AbstractMemPalaceEmbedder`](@ref). The default
[`DeterministicEmbedder`](@ref) hashes tokens into a fixed-dimensional
vector — useful for tests and reproducible benchmarks. For real
deployments use [`OllamaEmbedder`](@ref) (ships with the package) or
plug in any embedder that defines `embed(emb, text) -> Vector{Float64}`.

## AgentFramework integration

When both `MemPalace` and `AgentFramework` are loaded, the
`MemPalaceAgentFrameworkExt` weakdep extension activates and makes
available:

  * `MemPalaceMemoryStore <: AgentFramework.AbstractMemoryStore` — drop-in
    replacement for `InMemoryMemoryStore` / `SQLiteMemoryStore` etc.
    Plugs into the existing `MemoryContextProvider` so any agent that
    speaks the AF memory contract gets MemPalace storage for free.
  * `MemPalaceContextProvider <: AgentFramework.BaseContextProvider` —
    richer context provider that exposes wing/room scoping (read from
    `session.metadata`) and per-turn drawer ingestion.
"""
module MemPalace

using Dates
using HTTP
using JSON3
using LinearAlgebra
using Random
using SHA
using Statistics
using UUIDs
using Unicode

# ── Core types ───────────────────────────────────────────────────────────────
include("types.jl")

# ── Embedders ────────────────────────────────────────────────────────────────
include("embedder.jl")

# ── Topic / entity extraction ────────────────────────────────────────────────
include("extractor.jl")

# ── Backends ─────────────────────────────────────────────────────────────────
include("backends/base.jl")
include("backends/in_memory.jl")

# ── Hybrid search ────────────────────────────────────────────────────────────
include("searcher.jl")

# ── High-level palace ────────────────────────────────────────────────────────
include("palace.jl")

# ── Public API ───────────────────────────────────────────────────────────────
export Drawer, Closet, SearchHit, SearchResults
export AbstractPalaceBackend, InMemoryPalaceBackend
export add_drawer!, upsert_drawer!, add_closet!, upsert_closet!,
       get_drawer, query_drawers, query_closets,
       delete_drawers!, count_drawers, count_closets, clear!
export AbstractMemPalaceEmbedder, DeterministicEmbedder, OllamaEmbedder, embed
export build_closet_lines, extract_entities, extract_topics
export bm25_scores, hybrid_rank!
export Palace, mine_text!, mine_conversation!, search, wake_up

end # module MemPalace
