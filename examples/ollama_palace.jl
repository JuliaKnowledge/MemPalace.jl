# ollama_palace.jl
#
# Live MemPalace × Ollama vignette. Mirrors the corresponding Mem0.jl /
# Graphiti.jl Ollama smoke tests so all three memory backends in the
# AgentFramework.jl ecosystem have an end-to-end live example.
#
# Requires:
#   - A locally running Ollama (default http://localhost:11434)
#   - The embedding model `nomic-embed-text:latest` pulled
#       (`ollama pull nomic-embed-text`)
#
# Run with:
#   julia --project=. examples/ollama_palace.jl

using MemPalace
using Test

const MODEL = get(ENV, "MEMPALACE_OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
const BASE  = get(ENV, "OLLAMA_HOST", "http://localhost:11434")

@info "Embedder: $MODEL @ $BASE"

emb    = OllamaEmbedder(model=MODEL, base_url=BASE, dim=768)
palace = Palace(embedder=emb, chunk_chars=200, chunk_overlap=20)

mine_text!(palace, """
Alice loves Julia and works on the parser team. Her favourite colour is blue.
She has a cat named Whiskers and lives in Edinburgh.
""", wing="alice", room="profile", source_file="alice_profile")

mine_text!(palace, """
Bob recently moved to Seattle. He drinks green tea every morning.
Bob is on the analytics team and uses Python and R daily.
""", wing="bob", room="profile", source_file="bob_profile")

# Wing-scoped semantic recall
results = search(palace, "What pets does Alice have?", wing="alice")
@info "Alice pet search: $(length(results.hits)) hit(s)"
for (i, hit) in enumerate(results.hits)
    println("  [$i] ($(round(hit.score, digits=3))) $(first(hit.drawer.text, 80))…")
end

# Cross-wing wake-up over multiple cues
warm = wake_up(palace, ["What does Alice like?", "Where does Bob live?"]; per_query=2)
@info "wake_up: $(length(warm)) hit(s)"
for hit in warm
    println("  ($(round(hit.score, digits=3))) $(first(hit.drawer.text, 80))…")
end

@test length(results.hits) >= 1
@test any(occursin("Whiskers", h.drawer.text) for h in results.hits)
@test length(warm) >= 1

println("\nPASS — MemPalace × Ollama")
