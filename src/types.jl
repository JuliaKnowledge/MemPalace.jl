# Core types: Drawer (verbatim chunk) and Closet (topic pointer line bundle).

"""
    Drawer

Verbatim chunk of stored text plus its location in the palace.

# Fields

- `id::String`           — globally unique identifier (default UUID v4).
- `text::String`         — verbatim content. Never paraphrased or summarised.
- `wing::String`         — coarse entity scope (e.g. project name, person).
- `room::String`         — finer topic scope (e.g. "code", "design", "notes").
- `source_file::String`  — origin file path (`""` if mined from a string).
- `chunk_index::Int`     — position of this chunk within the source file.
- `source_mtime::Float64` — mtime of the source file at mine time, for
   freshness checks.
- `normalize_version::Int` — schema version of the normalisation pipeline
   (bumped when noise-stripping rules change).
- `created_at::DateTime` — UTC ingest time.
- `metadata::Dict{String, Any}` — backend-specific extra fields (kept
   round-trippable through JSON).
"""
Base.@kwdef mutable struct Drawer
    id::String = string(uuid4())
    text::String
    wing::String = "default"
    room::String = "general"
    source_file::String = ""
    chunk_index::Int = 0
    source_mtime::Float64 = 0.0
    normalize_version::Int = 2
    created_at::DateTime = now(UTC)
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function Base.show(io::IO, d::Drawer)
    preview = first(replace(d.text, '\n' => ' '), 48)
    print(io, "Drawer(\"", preview,
              length(d.text) > 48 ? "..." : "",
              "\", wing=", repr(d.wing),
              ", room=", repr(d.room), ")")
end

"""
    Closet

Compact pointer index built from one or more drawers in the same source.
Each closet packs a list of `topic|entities|→drawer_ids` lines (greedily
filled to ~`CLOSET_CHAR_LIMIT` characters per closet) so a single vector
search over closets surfaces the relevant set of drawer ids.

# Fields

- `id::String`           — closet id (typically `<base>_NN`).
- `text::String`         — newline-joined topic-pointer lines.
- `wing::String` / `room::String` / `source_file::String` — copied from
   the drawers it indexes.
- `drawer_ids::Vector{String}` — drawer ids referenced by any line.
- `created_at::DateTime`
- `metadata::Dict{String, Any}`
"""
Base.@kwdef mutable struct Closet
    id::String
    text::String
    wing::String = "default"
    room::String = "general"
    source_file::String = ""
    drawer_ids::Vector{String} = String[]
    created_at::DateTime = now(UTC)
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function Base.show(io::IO, c::Closet)
    n = length(c.drawer_ids)
    print(io, "Closet(\"", c.id,
              "\", ", n, " drawer", n == 1 ? "" : "s",
              ", wing=", repr(c.wing), ")")
end

# Schema constants — mirror the upstream Python module.

"Bump when normalisation rules change in a way that requires drawer rebuild."
const NORMALIZE_VERSION = 2

"Maximum characters of topic-pointer text packed into a single closet."
const CLOSET_CHAR_LIMIT = 1500

"How many characters of source content to scan when extracting topics/entities."
const CLOSET_EXTRACT_WINDOW = 5000
