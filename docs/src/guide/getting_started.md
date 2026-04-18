# Getting started with MemPalace.jl

MemPalace.jl is a Julia port of the
[MemPalace](https://github.com/MemPalace/mempalace) "verbatim memory
palace" — a local-first memory system organised as:

| Structure | What it is                                        |
|-----------|---------------------------------------------------|
| *wing*    | Entity scope (a user, a project, a session)       |
| *room*    | Topic scope within a wing (e.g. `bugs`, `notes`)  |
| *drawer*  | A verbatim chunk of text with metadata            |
| *closet*  | A packed topic-pointer index into several drawers |

Unlike vector-only stores, drawers are stored verbatim — you never get
a paraphrase back. Hybrid BM25 + cosine retrieval with a closet-based
boost combines lexical and semantic signals.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaKnowledge/MemPalace.jl")
```

## Thirty-second tour

```@example quickstart
using MemPalace

p = Palace(chunk_chars = 250, chunk_overlap = 30)

mine_text!(p, """
    # 2025-02-10  Alice
    - Fixed the parser bug that crashed on empty input.
    - Reviewed Bob's tokenizer PR.
    # 2025-02-12  Alice
    - Shipped v0.4.1 to staging.
    """;
    wing = "alice", room = "changelog",
    source_file = "changelog-alice.md")

mine_text!(p, """
    # 2025-02-10  Bob
    - Replaced the hand-rolled tokenizer with a regex state machine.
    """;
    wing = "bob", room = "changelog",
    source_file = "changelog-bob.md")

hits = search(p, "who fixed the parser bug?"; n_results = 3)
for h in hits
    println(round(h.score; digits=3), "  [", h.drawer.wing, "] ",
            first(h.drawer.text, 60))
end
```

Key points:

* `source_file` makes `mine_text!` idempotent — re-mining the same file
  deletes previous drawers before ingesting the new text.
* `wing = "alice"` filters retrieval to Alice's memories only:

```@example quickstart
alice_hits = search(p, "what shipped?"; wing = "alice", n_results = 3)
@assert all(h -> h.drawer.wing == "alice", alice_hits)
```

* `wake_up(p, probes)` runs a batch of canned probes at session start to
  pre-load salient facts:

```@example quickstart
bundle = wake_up(p, ["recent bug fixes", "deployment notes"];
                 per_query = 2, total = 5)
length(bundle.hits)
```

## Plugging in a real embedder

The default `DeterministicEmbedder` is perfect for tests (token-hash →
fixed dim, L2-normalised) but does not capture semantics. For real use,
point MemPalace at Ollama or any callable that returns a
`Vector{Float64}`:

```julia
using MemPalace

emb = OllamaEmbedder(model = "nomic-embed-text", base_url = "http://localhost:11434",
                     dim = 768)
p = Palace(embedder = emb)
```

Ollama's `/api/embed` endpoint must be reachable. Any embedder that
defines `embed(emb, text)::Vector{Float64}` works — implement the
method on your own struct and pass it in.

## Plugging in a different backend

The retrieval layer is behind [`AbstractPalaceBackend`](@ref). The
default [`InMemoryPalaceBackend`](@ref) is fine for < 100k drawers.
Persistent backends can be dropped in without touching the rest of the
pipeline; the contract is roughly:

* `add_drawer!`, `upsert_drawer!`, `get_drawer`
* `query_drawers(backend, q_emb; wing, room, n_results, max_distance)`
* `add_closet!`, `upsert_closet!`, `query_closets`
* `delete_drawers!`, `count_drawers`, `count_closets`, `clear!`

See `src/backends/in_memory.jl` for a reference implementation.

## Using MemPalace with AgentFramework.jl

Load both packages and the weakdep extension activates:

```julia
using AgentFramework, MemPalace

# Option A — drop-in AbstractMemoryStore (plays with MemoryContextProvider):
store = MemPalaceMemoryStore()
provider = MemoryContextProvider(store = store)

# Option B — native wing/room provider:
native = MemPalaceContextProvider(Palace(); room = "profile")
```

See [`examples/agentframework_ext.jl`](https://github.com/JuliaKnowledge/MemPalace.jl/blob/main/examples/agentframework_ext.jl)
for a runnable end-to-end walkthrough of the AgentFramework extension.

## Next steps

* Run [`examples/quickstart.jl`](https://github.com/JuliaKnowledge/MemPalace.jl/blob/main/examples/quickstart.jl)
  and [`examples/agentframework_ext.jl`](https://github.com/JuliaKnowledge/MemPalace.jl/blob/main/examples/agentframework_ext.jl).
* Read the [AgentFramework Memory Backends Compared](https://juliaknowledge.github.io/AgentFramework.jl/vignettes/memory_backends_compared/)
  vignette to see when to pick MemPalace vs Mem0 vs Graphiti.
