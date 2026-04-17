# In-memory backend. Brute-force cosine over a Vector{Float64} per record;
# fast enough for tests, demos, and palaces up to a few thousand drawers.

"""
    InMemoryPalaceBackend() <: AbstractPalaceBackend

Pure-Julia, brute-force backend. Stores drawers and closets in
parallel `Dict{String, ...}` maps with their embeddings alongside.
Queries scan all records and compute cosine similarity (≡ dot product
when both vectors are L2-normalised, which the package's embedders do).

Suitable for unit tests, examples, and small palaces (≤ ~10k drawers).
For larger workloads, plug in a backend that wraps SQLite-VSS, Chroma,
Qdrant, etc.
"""
mutable struct InMemoryPalaceBackend <: AbstractPalaceBackend
    drawers::Dict{String, Drawer}
    drawer_emb::Dict{String, Vector{Float64}}
    closets::Dict{String, Closet}
    closet_emb::Dict{String, Vector{Float64}}
end

InMemoryPalaceBackend() = InMemoryPalaceBackend(
    Dict{String, Drawer}(),
    Dict{String, Vector{Float64}}(),
    Dict{String, Closet}(),
    Dict{String, Vector{Float64}}(),
)

function Base.show(io::IO, b::InMemoryPalaceBackend)
    print(io, "InMemoryPalaceBackend(",
              length(b.drawers), " drawers, ",
              length(b.closets), " closets)")
end

# ── Drawers ──────────────────────────────────────────────────────────────────

function add_drawer!(b::InMemoryPalaceBackend, drawer::Drawer, embedding::Vector{Float64})
    haskey(b.drawers, drawer.id) && error("drawer id $(drawer.id) already exists")
    b.drawers[drawer.id] = drawer
    b.drawer_emb[drawer.id] = embedding
    return drawer
end

function upsert_drawer!(b::InMemoryPalaceBackend, drawer::Drawer, embedding::Vector{Float64})
    b.drawers[drawer.id] = drawer
    b.drawer_emb[drawer.id] = embedding
    return drawer
end

get_drawer(b::InMemoryPalaceBackend, id::AbstractString) = get(b.drawers, String(id), nothing)

count_drawers(b::InMemoryPalaceBackend)::Int = length(b.drawers)

# ── Closets ──────────────────────────────────────────────────────────────────

function add_closet!(b::InMemoryPalaceBackend, closet::Closet, embedding::Vector{Float64})
    haskey(b.closets, closet.id) && error("closet id $(closet.id) already exists")
    b.closets[closet.id] = closet
    b.closet_emb[closet.id] = embedding
    return closet
end

function upsert_closet!(b::InMemoryPalaceBackend, closet::Closet, embedding::Vector{Float64})
    b.closets[closet.id] = closet
    b.closet_emb[closet.id] = embedding
    return closet
end

count_closets(b::InMemoryPalaceBackend)::Int = length(b.closets)

# ── Querying ─────────────────────────────────────────────────────────────────

# Cosine similarity assuming embeddings may not be unit-normalised. When
# both inputs are unit vectors this collapses to a dot product, but we
# stay defensive so external embedders that don't normalise still work.
function _cosine(a::Vector{Float64}, b::Vector{Float64})::Float64
    isempty(a) && return 0.0
    isempty(b) && return 0.0
    na = norm(a); nb = norm(b)
    (na == 0.0 || nb == 0.0) && return 0.0
    return dot(a, b) / (na * nb)
end

function _drawer_passes_filter(d::Drawer; wing, room, source_file)::Bool
    wing !== nothing && d.wing != wing && return false
    room !== nothing && d.room != room && return false
    source_file !== nothing && d.source_file != source_file && return false
    return true
end

"""
    query_drawers(b, query_embedding; wing=nothing, room=nothing,
                  source_file=nothing, n_results=5, max_distance=2.0)

Return up to `n_results` `(drawer, distance)` pairs sorted by ascending
cosine distance (`1 - similarity`). Optional filters scope by wing,
room, or source file. Distances above `max_distance` (cosine distance is
in `[0, 2]`, 0 = identical) are filtered out.
"""
function query_drawers(b::InMemoryPalaceBackend,
                       query_embedding::Vector{Float64};
                       wing::Union{Nothing, AbstractString} = nothing,
                       room::Union{Nothing, AbstractString} = nothing,
                       source_file::Union{Nothing, AbstractString} = nothing,
                       n_results::Int = 5,
                       max_distance::Float64 = 2.0)::Vector{Tuple{Drawer, Float64}}
    scored = Tuple{Drawer, Float64}[]
    for (id, d) in b.drawers
        _drawer_passes_filter(d; wing, room, source_file) || continue
        emb = b.drawer_emb[id]
        sim = _cosine(query_embedding, emb)
        dist = 1.0 - sim
        dist > max_distance && continue
        push!(scored, (d, dist))
    end
    sort!(scored; by = x -> x[2])
    return scored[1:min(n_results, length(scored))]
end

"""
    query_closets(b, query_embedding; wing=nothing, room=nothing,
                  n_results=5)

Return up to `n_results` `(closet, distance)` pairs.
"""
function query_closets(b::InMemoryPalaceBackend,
                       query_embedding::Vector{Float64};
                       wing::Union{Nothing, AbstractString} = nothing,
                       room::Union{Nothing, AbstractString} = nothing,
                       n_results::Int = 5)::Vector{Tuple{Closet, Float64}}
    scored = Tuple{Closet, Float64}[]
    for (id, c) in b.closets
        wing !== nothing && c.wing != wing && continue
        room !== nothing && c.room != room && continue
        emb = b.closet_emb[id]
        push!(scored, (c, 1.0 - _cosine(query_embedding, emb)))
    end
    sort!(scored; by = x -> x[2])
    return scored[1:min(n_results, length(scored))]
end

"""
    delete_drawers!(b; source_file=nothing, ids=nothing) -> Int

Delete drawers matching the given `source_file` or any id in `ids`.
At least one of the kwargs must be given. Returns the number of drawers
removed.
"""
function delete_drawers!(b::InMemoryPalaceBackend;
                         source_file::Union{Nothing, AbstractString} = nothing,
                         ids::Union{Nothing, Vector{<:AbstractString}} = nothing)::Int
    source_file === nothing && ids === nothing &&
        throw(ArgumentError("supply source_file or ids"))
    to_remove = String[]
    if source_file !== nothing
        for (id, d) in b.drawers
            d.source_file == source_file && push!(to_remove, id)
        end
    end
    if ids !== nothing
        for id in ids
            haskey(b.drawers, String(id)) && push!(to_remove, String(id))
        end
    end
    unique!(to_remove)
    for id in to_remove
        delete!(b.drawers, id)
        delete!(b.drawer_emb, id)
    end
    return length(to_remove)
end

function clear!(b::InMemoryPalaceBackend)
    empty!(b.drawers); empty!(b.drawer_emb)
    empty!(b.closets); empty!(b.closet_emb)
    return b
end
