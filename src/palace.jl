# High-level Palace orchestrator. Combines a backend, an embedder, and the
# mining + retrieval pipelines into a single object the user interacts with.

"""
    Palace(; backend = InMemoryPalaceBackend(),
            embedder = DeterministicEmbedder())

The top-level MemPalace object. Bundles a backend (any
`AbstractPalaceBackend`) with an embedder (any
`AbstractMemPalaceEmbedder`) so the rest of the API doesn't need to
thread them through every call.

# Example

```julia
using MemPalace

p = Palace()
mine_text!(p, "Alice fixed the parser bug on Monday.";
           wing="alice", room="bugs", source_file="notes.md")
hits = search(p, "what did Alice fix?"; wing="alice")
for h in hits
    println(h.score, "  ", h.drawer.text)
end
```
"""
Base.@kwdef mutable struct Palace{B <: AbstractPalaceBackend,
                                  E <: AbstractMemPalaceEmbedder}
    backend::B = InMemoryPalaceBackend()
    embedder::E = DeterministicEmbedder()
    chunk_chars::Int = 1500
    chunk_overlap::Int = 100
    default_wing::String = "default"
    default_room::String = "general"
end

function Base.show(io::IO, p::Palace)
    print(io, "Palace(backend=", p.backend,
              ", embedder=", typeof(p.embedder).name.name, ")")
end

# ── Chunking ─────────────────────────────────────────────────────────────────

"""
    _chunk_text(text, size, overlap) -> Vector{String}

Split `text` into chunks of approximately `size` characters with
`overlap` characters carried into the next chunk. Splits on character
boundaries (not bytes) so multi-byte UTF-8 stays intact.
"""
function _chunk_text(text::AbstractString, size::Int, overlap::Int)::Vector{String}
    s = String(text)
    isempty(s) && return String[]
    chars = collect(s)
    n = length(chars)
    n <= size && return [s]
    chunks = String[]
    step = max(size - overlap, 1)
    i = 1
    while i <= n
        j = min(i + size - 1, n)
        push!(chunks, String(chars[i:j]))
        j == n && break
        i += step
    end
    return chunks
end

# ── Mining ───────────────────────────────────────────────────────────────────

"""
    mine_text!(palace, text;
               wing=palace.default_wing, room=palace.default_room,
               source_file="", source_mtime=0.0,
               build_closets=true) -> Vector{Drawer}

Ingest `text` into the palace. Splits into drawer-sized chunks, embeds
each, stores them, and (when `build_closets=true`) builds closet pointer
lines and packs them into ≤`CLOSET_CHAR_LIMIT`-char closets.

If `source_file` is non-empty, any drawers/closets previously mined from
that file are deleted first so the new content fully replaces the old.
Returns the list of drawers written.
"""
function mine_text!(p::Palace, text::AbstractString;
                    wing::AbstractString = p.default_wing,
                    room::AbstractString = p.default_room,
                    source_file::AbstractString = "",
                    source_mtime::Float64 = 0.0,
                    build_closets::Bool = true)::Vector{Drawer}
    if !isempty(source_file)
        delete_drawers!(p.backend; source_file = source_file)
    end

    chunks = _chunk_text(text, p.chunk_chars, p.chunk_overlap)
    drawers = Drawer[]
    for (i, chunk) in enumerate(chunks)
        d = Drawer(
            text = chunk,
            wing = String(wing),
            room = String(room),
            source_file = String(source_file),
            chunk_index = i - 1,
            source_mtime = source_mtime,
        )
        emb = embed(p.embedder, chunk)
        add_drawer!(p.backend, d, emb)
        push!(drawers, d)
    end

    if build_closets && !isempty(drawers)
        ids = [d.id for d in drawers]
        lines = build_closet_lines(source_file, ids, text, wing, room)
        _pack_and_store_closets!(p, lines, source_file, wing, room)
    end

    return drawers
end

"""
    mine_conversation!(palace, messages; wing, room, source_file="")
        -> Vector{Drawer}

Ingest a conversation turn-by-turn. Each `messages[i]` should be either
a `Dict("role" => ..., "content" => ...)` or any object with `.role` and
`.content` fields. Each non-empty turn becomes one drawer (no further
chunking) so utterances can be retrieved verbatim.
"""
function mine_conversation!(p::Palace, messages;
                            wing::AbstractString = p.default_wing,
                            room::AbstractString = p.default_room,
                            source_file::AbstractString = "")::Vector{Drawer}
    if !isempty(source_file)
        delete_drawers!(p.backend; source_file = source_file)
    end
    drawers = Drawer[]
    for (i, m) in enumerate(messages)
        role = _msg_field(m, :role, "user")
        content = _msg_field(m, :content, "")
        text = strip(String(content))
        isempty(text) && continue
        d = Drawer(
            text = string(role, ": ", text),
            wing = String(wing), room = String(room),
            source_file = String(source_file),
            chunk_index = i - 1,
            metadata = Dict{String, Any}("role" => String(role)),
        )
        add_drawer!(p.backend, d, embed(p.embedder, text))
        push!(drawers, d)
    end
    if !isempty(drawers)
        full = join((d.text for d in drawers), "\n")
        ids = [d.id for d in drawers]
        lines = build_closet_lines(source_file, ids, full, wing, room)
        _pack_and_store_closets!(p, lines, source_file, wing, room)
    end
    return drawers
end

function _msg_field(m, sym::Symbol, default)
    if m isa AbstractDict
        return get(m, String(sym), get(m, sym, default))
    end
    return hasproperty(m, sym) ? getproperty(m, sym) : default
end

function _pack_and_store_closets!(p::Palace, lines::Vector{String},
                                   source_file::AbstractString,
                                   wing::AbstractString, room::AbstractString)
    base = isempty(source_file) ?
           string("closet_", first(string(uuid4()), 8)) :
           string("closet_", bytes2hex(sha256(source_file))[1:12])
    closet_num = 1
    current = String[]
    cur_chars = 0

    function _flush()
        isempty(current) && return
        cid = string(base, "_", lpad(closet_num, 2, '0'))
        text = join(current, "\n")
        emb = embed(p.embedder, text)
        c = Closet(
            id = cid, text = text,
            wing = String(wing), room = String(room),
            source_file = String(source_file),
            drawer_ids = unique!(extract_drawer_ids_from_closet(text)),
        )
        upsert_closet!(p.backend, c, emb)
    end

    for line in lines
        l = length(line)
        if cur_chars > 0 && cur_chars + l + 1 > CLOSET_CHAR_LIMIT
            _flush()
            closet_num += 1
            current = String[]
            cur_chars = 0
        end
        push!(current, line)
        cur_chars += l + 1
    end
    _flush()
    return nothing
end

# ── Retrieval ────────────────────────────────────────────────────────────────

"""
    search(palace, query;
           wing=nothing, room=nothing,
           n_results=5, max_distance=2.0,
           closet_boost_step=0.05,
           vector_weight=0.6, bm25_weight=0.4) -> SearchResults

Hybrid retrieval over the palace.

1. Embed the query.
2. Over-fetch `n_results * 3` drawers by cosine distance (the *floor*).
3. Pull `n_results * 2` closets and build a `source_file → boost` map.
4. Apply the closet boost (`closet_boost_step` per rank earlier than the
   drawer's own rank).
5. Re-rank by a convex combination of cosine similarity and BM25 over
   the candidate set.

Returns the top `n_results` `SearchHit`s. Closets are a *signal*, never
a *gate* — direct drawer search is always the baseline (matches upstream
"weak-closets regression" fix).
"""
function search(p::Palace, query::AbstractString;
                wing::Union{Nothing, AbstractString} = nothing,
                room::Union{Nothing, AbstractString} = nothing,
                n_results::Int = 5,
                max_distance::Float64 = 2.0,
                closet_boost_step::Float64 = 0.05,
                vector_weight::Float64 = 0.6,
                bm25_weight::Float64 = 0.4)::SearchResults
    qstr = String(query)
    qemb = embed(p.embedder, qstr)

    drawer_pairs = query_drawers(p.backend, qemb;
                                  wing = wing, room = room,
                                  n_results = n_results * 3,
                                  max_distance = max_distance)

    # Closet boost lookup: source_file => boost (higher = better).
    closet_boost = Dict{String, Float64}()
    closet_preview = Dict{String, String}()
    closet_pairs = query_closets(p.backend, qemb;
                                  wing = wing, room = room,
                                  n_results = n_results * 2)
    for (rank, (c, _)) in enumerate(closet_pairs)
        src = c.source_file
        isempty(src) && continue
        haskey(closet_boost, src) && continue
        boost = max(0.0, closet_boost_step * (length(closet_pairs) - rank + 1))
        closet_boost[src] = boost
        closet_preview[src] = first(c.text, 200)
    end

    hits = SearchHit[]
    for (d, dist) in drawer_pairs
        b = get(closet_boost, d.source_file, 0.0)
        prev = get(closet_preview, d.source_file, nothing)
        push!(hits, SearchHit(
            drawer = d, score = 0.0, distance = dist,
            closet_boost = b, closet_preview = prev,
        ))
    end

    hybrid_rank!(hits, qstr;
                  vector_weight = vector_weight, bm25_weight = bm25_weight)

    return SearchResults(qstr, hits[1:min(n_results, length(hits))])
end

"""
    wake_up(palace, queries; per_query=3, total=10) -> SearchResults

Convenience: run several "wake-up" queries, dedupe by drawer id, and
return up to `total` hits ranked by best per-query score. Mirrors the
upstream `mempalace wake-up` CLI command which loads context for a new
session from a small bundle of canned probes.
"""
function wake_up(p::Palace, queries::Vector{<:AbstractString};
                 per_query::Int = 3, total::Int = 10)::SearchResults
    seen = Dict{String, SearchHit}()
    for q in queries
        for h in search(p, q; n_results = per_query)
            existing = get(seen, h.drawer.id, nothing)
            if existing === nothing || h.score > existing.score
                seen[h.drawer.id] = h
            end
        end
    end
    hits = collect(values(seen))
    sort!(hits; by = h -> -h.score)
    return SearchResults(join(queries, " | "), hits[1:min(total, length(hits))])
end
