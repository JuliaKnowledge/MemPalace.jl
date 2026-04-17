# MemPalace × AgentFramework extension vignette
#
# Narrative: a support agent needs persistent, scope-isolated memory per
# end-user. We show the two ways MemPalace plugs into AgentFramework:
#
#   (1) `MemPalaceMemoryStore` — drop-in `AbstractMemoryStore`. Any code
#       that already uses `MemoryContextProvider` (the built-in generic
#       vector-memory provider) gets MemPalace storage for free.
#
#   (2) `MemPalaceContextProvider` — a richer native provider that honors
#       MemPalace's wing/room scoping (read from `session.metadata`) and
#       can optionally ingest every turn as a fresh drawer.
#
# Both paths are exercised below against the default DeterministicEmbedder
# so the vignette runs offline.
#
# Run:  julia --project=. examples/agentframework_ext.jl

using MemPalace
using AgentFramework
using Test

const Ext = Base.get_extension(MemPalace, :MemPalaceAgentFrameworkExt)
@assert Ext !== nothing "MemPalaceAgentFrameworkExt did not load"

println("── Part 1: MemPalaceMemoryStore as a drop-in AbstractMemoryStore ──")

store = Ext.MemPalaceMemoryStore(palace = Palace(embedder = DeterministicEmbedder(128)))

# Seed memories for two different users — scope keeps them isolated.
add_memories!(store, [
    MemoryRecord(scope="alice", role=:user,
                  content="I prefer dark mode across all tools."),
    MemoryRecord(scope="alice", role=:assistant,
                  content="Acknowledged — dark mode preference saved."),
    MemoryRecord(scope="bob",   role=:user,
                  content="Please always cite sources in your answers."),
])

# Re-use the *generic* MemoryContextProvider — the AF abstraction. This
# is the payoff: MemPalace speaks the AbstractMemoryStore contract so any
# generic provider works untouched.
provider = MemoryContextProvider(store = store, max_results = 3)
@test provider.store === store

alice_mem = search_memories(store, "how do I like my UI?"; scope="alice")
@test !isempty(alice_mem)
@test occursin("dark", first(alice_mem).record.content)
@test all(r -> r.record.scope == "alice", alice_mem)

println("  alice recall: ", first(alice_mem).record.content)
println("  store: ", store)

println("\n── Part 2: MemPalaceContextProvider with wing/room scoping ──")

p = Palace(embedder = DeterministicEmbedder(128),
           chunk_chars = 240, chunk_overlap = 20)

# Pre-seed per-wing context — "profile" for long-lived facts, "chat" for
# conversational turns.
mine_text!(p, "Alice is on the platform team and prefers dark mode UIs.";
           wing="alice", room="profile")
mine_text!(p, "Bob leads the deployment squad and asks for cited sources.";
           wing="bob",   room="profile")

# The provider reads the wing from session.user_id (falls back to
# session.metadata["memory_scope"] if set) and pins the room here.
provider2 = Ext.MemPalaceContextProvider(p;
    room      = "profile",
    n_results = 3,
    store     = false,  # this example doesn't write back
)

# Simulate an agent turn for Alice without actually calling an LLM — we
# just run the pre-hook and inspect the injected context messages.
# Note: providers push retrieved context into `ctx.context_messages` keyed
# by their `source_id`, NOT into `ctx.input_messages` — the agent runner
# merges both when building the LLM prompt.
sess_alice = AgentSession(user_id = "alice")
ctx_alice  = SessionContext()
push!(ctx_alice.input_messages, Message(:user, "remind me about my UI preference"))
state = Dict{String, Any}()
AgentFramework.before_run!(provider2, nothing, sess_alice, ctx_alice, state)

@test haskey(ctx_alice.context_messages, "mempalace")
injected = join([m.text for m in ctx_alice.context_messages["mempalace"]], "\n")
@info "Injected context for alice" snippet=first(injected, 160)
@test occursin("dark", lowercase(injected))
@test !occursin("deployment", lowercase(injected))

# Switching user — Bob's memories now, Alice's stay out.
sess_bob = AgentSession(user_id = "bob")
ctx_bob  = SessionContext()
push!(ctx_bob.input_messages, Message(:user, "anything I should remember about citations?"))
AgentFramework.before_run!(provider2, nothing, sess_bob, ctx_bob, state)
injected_bob = join([m.text for m in ctx_bob.context_messages["mempalace"]], "\n")
@test occursin("cite", lowercase(injected_bob)) || occursin("cited", lowercase(injected_bob))
@test !occursin("dark mode", lowercase(injected_bob))

println("\nPASS — MemPalace × AgentFramework extension vignette")
