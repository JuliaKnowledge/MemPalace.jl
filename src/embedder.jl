# Embedder abstraction for MemPalace.
# Mirrors the pluggable-embedder design used in Mem0.jl and Graphiti.jl.

"""
    AbstractMemPalaceEmbedder

Subtypes implement `embed(emb, text::AbstractString) -> Vector{Float64}`.
The embedding dimensionality is exposed via `Base.length(emb)::Int`.

This abstraction matches the upstream MemPalace `BaseCollection` design
where embedding generation is delegated to the backing store (ChromaDB)
rather than being part of the public API. In Julia we make it explicit so
that callers can swap embedders independently of the backend.
"""
abstract type AbstractMemPalaceEmbedder end

"Generic dispatch hook — backends call this to vectorise drawer text."
function embed end

# ── DeterministicEmbedder ────────────────────────────────────────────────────

"""
    DeterministicEmbedder(dim::Int = 384) <: AbstractMemPalaceEmbedder

Pure-Julia deterministic embedder built from a token-hash bag-of-words.
Produces an L2-normalised `Vector{Float64}` of length `dim`.

Uses SHA-256 over each whitespace-stripped lowercase token to pick a
bucket, then accumulates a 1.0 contribution per occurrence. The result
is L2-normalised so cosine similarity == dot product.

Useful for tests, benchmarks, and offline reproducibility — *not* a
substitute for a real semantic embedder in production.
"""
struct DeterministicEmbedder <: AbstractMemPalaceEmbedder
    dim::Int
end
DeterministicEmbedder() = DeterministicEmbedder(384)

Base.length(e::DeterministicEmbedder) = e.dim

const _TOKEN_RX = r"\w{2,}"

function embed(emb::DeterministicEmbedder, text::AbstractString)::Vector{Float64}
    v = zeros(Float64, emb.dim)
    isempty(text) && return v
    lower = lowercase(String(text))
    for m in eachmatch(_TOKEN_RX, lower)
        tok = m.match
        h = sha256(tok)
        # Take the first 8 bytes as a UInt64; bucket modulo dim.
        bucket = mod(reinterpret(UInt64, h[1:8])[1], emb.dim) + 1
        v[bucket] += 1.0
        # Add a sign bit from byte 9 to spread mass into +/-, helping
        # discrimination at small dims.
        sgn = iseven(h[9]) ? 1.0 : -1.0
        v[bucket] += sgn * 0.25
    end
    n = norm(v)
    n > 0 && (v ./= n)
    return v
end

# ── OllamaEmbedder ───────────────────────────────────────────────────────────

"""
    OllamaEmbedder(; model="nomic-embed-text", base_url="http://localhost:11434", dim=768)

Embedder that calls a local Ollama server's `/api/embed` endpoint.
Mirrors the embedder design in `Mem0.jl/src/embeddings/ollama.jl`.
"""
Base.@kwdef mutable struct OllamaEmbedder <: AbstractMemPalaceEmbedder
    model::String = "nomic-embed-text"
    base_url::String = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
    dim::Int = 768
end

Base.length(e::OllamaEmbedder) = e.dim

function embed(emb::OllamaEmbedder, text::AbstractString)::Vector{Float64}
    isempty(text) && return zeros(Float64, emb.dim)
    body = Dict{String, Any}(
        "model" => emb.model,
        "input" => String(text),
    )
    resp = HTTP.post(
        "$(emb.base_url)/api/embed",
        ["Content-Type" => "application/json"],
        JSON3.write(body);
        status_exception = false,
    )
    if resp.status != 200
        error("MemPalace.OllamaEmbedder: API request failed ($(resp.status)): $(String(resp.body))")
    end
    data = JSON3.read(String(resp.body), Dict{String, Any})
    embeddings = data["embeddings"]
    return Float64.(embeddings[1])
end
