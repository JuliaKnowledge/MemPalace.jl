# Topic / entity extraction. Ports the regex-based candidate-entity and
# topic-line extraction from upstream `mempalace/palace.py`.

const _ENTITY_STOPLIST = Set{String}([
    "The", "This", "That", "These", "Those",
    "When", "Where", "What", "Why", "Who", "Which", "How",
    "After", "Before", "Then", "Now", "Here", "There",
    "And", "But", "Or", "Yet", "So", "If", "Else",
    "Yes", "No", "Maybe", "Okay",
    "User", "Assistant", "System", "Tool",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
])

# Capitalised word followed by optional capitalised continuation.
# Equivalent to upstream's default English `candidate_patterns`.
const _ENTITY_CANDIDATE_RX = r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b"

# Action verbs that anchor topic lines.
const _TOPIC_VERB_RX = r"(?i)(?:built|fixed|wrote|added|pushed|tested|created|decided|migrated|reviewed|deployed|configured|removed|updated)\s+[\w\s]{3,40}"

# Markdown section headers.
const _HEADER_RX = r"(?m)^#{1,3}\s+(.{5,60})$"

# Quoted strings.
const _QUOTE_RX = r"\"([^\"]{15,150})\""

"""
    extract_entities(text; min_count=2, top_k=5) -> Vector{String}

Find proper-noun candidate words in `text`, drop a stoplist of
sentence-starters / fillers, keep only those that occur ≥ `min_count`
times, and return the top `top_k` by frequency.
"""
function extract_entities(text::AbstractString; min_count::Int = 2, top_k::Int = 5)::Vector{String}
    window = first(String(text), CLOSET_EXTRACT_WINDOW)
    counts = Dict{String, Int}()
    for m in eachmatch(_ENTITY_CANDIDATE_RX, window)
        word = m.match
        first(word) in _ENTITY_STOPLIST && continue
        word in _ENTITY_STOPLIST && continue
        counts[word] = get(counts, word, 0) + 1
    end
    entities = [w for (w, c) in counts if c >= min_count]
    sort!(entities; by = w -> -counts[w])
    return entities[1:min(top_k, length(entities))]
end

"""
    extract_topics(text; top_k=12) -> Vector{String}

Extract action-verb topic phrases and markdown headers from `text`.
Lower-cased, deduplicated, capped at `top_k`.
"""
function extract_topics(text::AbstractString; top_k::Int = 12)::Vector{String}
    window = first(String(text), CLOSET_EXTRACT_WINDOW)
    raw = String[]
    for m in eachmatch(_TOPIC_VERB_RX, window)
        push!(raw, m.match)
    end
    for m in eachmatch(_HEADER_RX, window)
        push!(raw, m.captures[1])
    end
    seen = Set{String}()
    out = String[]
    for r in raw
        s = lowercase(strip(r))
        isempty(s) && continue
        s in seen && continue
        push!(seen, s)
        push!(out, s)
    end
    return out[1:min(top_k, length(out))]
end

function _extract_quotes(text::AbstractString; top_k::Int = 3)::Vector{String}
    window = first(String(text), CLOSET_EXTRACT_WINDOW)
    out = String[]
    for m in eachmatch(_QUOTE_RX, window)
        push!(out, m.captures[1])
        length(out) >= top_k && break
    end
    return out
end

"""
    build_closet_lines(source_file, drawer_ids, content, wing, room) -> Vector{String}

Build the list of `topic|entities|→drawer_ids` pointer lines that index
the supplied `content`. Each line is atomic and never split across
closets. If neither topics nor quotes are extractable, falls back to a
single `wing/room/<file_stem>` line so every drawer is still findable.
"""
function build_closet_lines(source_file::AbstractString,
                            drawer_ids::Vector{String},
                            content::AbstractString,
                            wing::AbstractString,
                            room::AbstractString)::Vector{String}
    drawer_ref = join(first(drawer_ids, min(3, length(drawer_ids))), ",")
    entities = extract_entities(content)
    topics = extract_topics(content)
    quotes = _extract_quotes(content)
    entity_str = isempty(entities) ? "" : join(entities, ";")

    lines = String[]
    for t in topics
        push!(lines, string(t, "|", entity_str, "|→", drawer_ref))
    end
    for q in quotes
        push!(lines, string("\"", q, "\"|", entity_str, "|→", drawer_ref))
    end
    if isempty(lines)
        stem = splitext(basename(String(source_file)))[1]
        stem = first(isempty(stem) ? "drawer" : stem, 40)
        push!(lines, string(wing, "/", room, "/", stem, "|", entity_str, "|→", drawer_ref))
    end
    return lines
end

"""
    extract_drawer_ids_from_closet(closet_text) -> Vector{String}

Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.
Preserves first-seen order and deduplicates.
"""
function extract_drawer_ids_from_closet(closet_text::AbstractString)::Vector{String}
    seen = Set{String}()
    out = String[]
    for m in eachmatch(r"→([\w\-,]+)", String(closet_text))
        for did in split(m.captures[1], ",")
            d = strip(did)
            isempty(d) && continue
            d in seen && continue
            push!(seen, String(d))
            push!(out, String(d))
        end
    end
    return out
end
