# Hybrid retrieval: vector cosine (the floor) + BM25 reranking + closet boost.
# Ports `mempalace/searcher.py`'s `_bm25_scores`, `_hybrid_rank`, and
# `search_memories` to Julia.

"""
    SearchHit

A single retrieval hit returned by `search`.

# Fields
- `drawer::Drawer`        — the matched drawer.
- `score::Float64`        — final hybrid rank score (higher = better).
- `distance::Float64`     — vector cosine distance from the query (`1 - sim`).
- `bm25::Float64`         — raw BM25 score over the candidate set.
- `closet_boost::Float64` — additive boost contributed by an aligned closet.
- `closet_preview::Union{Nothing,String}} — first 200 chars of the
   contributing closet, when one exists.
"""
Base.@kwdef struct SearchHit
    drawer::Drawer
    score::Float64
    distance::Float64
    bm25::Float64 = 0.0
    closet_boost::Float64 = 0.0
    closet_preview::Union{Nothing, String} = nothing
end

function Base.show(io::IO, h::SearchHit)
    print(io, "SearchHit(score=", round(h.score; digits=3),
              ", dist=", round(h.distance; digits=3),
              ", ", h.drawer, ")")
end

"""
    SearchResults

Wraps a vector of `SearchHit`s plus the original query for context.
Implements `Base.iterate`, `Base.length`, and `Base.getindex` so callers
can treat it like a `Vector{SearchHit}`.
"""
struct SearchResults
    query::String
    hits::Vector{SearchHit}
end

Base.length(r::SearchResults) = length(r.hits)
Base.iterate(r::SearchResults, args...) = iterate(r.hits, args...)
Base.getindex(r::SearchResults, i) = r.hits[i]
Base.isempty(r::SearchResults) = isempty(r.hits)

function Base.show(io::IO, r::SearchResults)
    print(io, "SearchResults(", length(r.hits), " hits, query=", repr(r.query), ")")
end

# ── BM25 ─────────────────────────────────────────────────────────────────────

const _SEARCH_TOKEN_RX = r"\w{2,}"

function _tokenize(text::AbstractString)::Vector{String}
    return [m.match for m in eachmatch(_SEARCH_TOKEN_RX, lowercase(String(text)))]
end

"""
    bm25_scores(query, documents; k1=1.5, b=0.75) -> Vector{Float64}

Okapi-BM25 with corpus-relative IDF over the supplied candidate set.
The Lucene smoothed IDF formula `log((N - df + 0.5) / (df + 0.5) + 1)`
is non-negative, suitable for re-ranking a small candidate window.
"""
function bm25_scores(query::AbstractString,
                     documents::Vector{<:AbstractString};
                     k1::Float64 = 1.5, b::Float64 = 0.75)::Vector{Float64}
    n = length(documents)
    n == 0 && return Float64[]
    qterms = Set(_tokenize(query))
    isempty(qterms) && return zeros(Float64, n)

    tokenised = [_tokenize(d) for d in documents]
    doc_lens = [length(t) for t in tokenised]
    avgdl = sum(doc_lens) / max(n, 1)
    avgdl == 0.0 && return zeros(Float64, n)

    df = Dict{String, Int}()
    for term in qterms; df[term] = 0; end
    for toks in tokenised
        seen = Set(toks) ∩ qterms
        for term in seen
            df[term] += 1
        end
    end

    idf = Dict{String, Float64}(
        term => log((n - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in qterms
    )

    scores = zeros(Float64, n)
    for (i, (toks, dl)) in enumerate(zip(tokenised, doc_lens))
        dl == 0 && continue
        tf = Dict{String, Int}()
        for t in toks
            t in qterms || continue
            tf[t] = get(tf, t, 0) + 1
        end
        s = 0.0
        for (term, freq) in tf
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            s += idf[term] * num / den
        end
        scores[i] = s
    end
    return scores
end

"""
    hybrid_rank!(hits::Vector{SearchHit}, query;
                 vector_weight=0.6, bm25_weight=0.4) -> Vector{SearchHit}

Reorder `hits` in place by a convex combination of vector similarity
(`max(0, 1 - distance)`) and min-max-normalised BM25 score over the
candidate set. Returns the same vector for chaining.
"""
function hybrid_rank!(hits::Vector{SearchHit}, query::AbstractString;
                      vector_weight::Float64 = 0.6,
                      bm25_weight::Float64 = 0.4)::Vector{SearchHit}
    isempty(hits) && return hits
    docs = [h.drawer.text for h in hits]
    raw = bm25_scores(query, docs)
    mx = maximum(raw; init = 0.0)
    norm_bm25 = mx > 0 ? raw ./ mx : zeros(Float64, length(raw))

    rebuilt = Vector{SearchHit}(undef, length(hits))
    for (i, h) in enumerate(hits)
        vec_sim = max(0.0, 1.0 - h.distance)
        score = vector_weight * vec_sim + bm25_weight * norm_bm25[i] + h.closet_boost
        rebuilt[i] = SearchHit(
            drawer = h.drawer,
            score = score,
            distance = h.distance,
            bm25 = raw[i],
            closet_boost = h.closet_boost,
            closet_preview = h.closet_preview,
        )
    end
    sort!(rebuilt; by = h -> -h.score)
    hits .= rebuilt
    return hits
end
