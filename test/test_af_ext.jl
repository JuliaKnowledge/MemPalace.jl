using MemPalace
using AgentFramework
using Test

const Ext = Base.get_extension(MemPalace, :MemPalaceAgentFrameworkExt)

@testset "MemPalaceAgentFrameworkExt loaded" begin
    @test Ext !== nothing
end

@testset "MemPalaceMemoryStore <: AbstractMemoryStore" begin
    p = Palace(embedder = DeterministicEmbedder(64))
    store = Ext.MemPalaceMemoryStore(palace = p)
    @test store isa AbstractMemoryStore

    rec1 = MemoryRecord(scope="alice", role=:user,
                         content="favorite color is blue")
    rec2 = MemoryRecord(scope="alice", role=:assistant,
                         content="noted: alice likes blue")
    rec3 = MemoryRecord(scope="bob",   role=:user,
                         content="bob owns the deployment")
    add_memories!(store, [rec1, rec2, rec3])
    @test occursin("3 drawers", sprint(show, store))

    # Search scoped to alice should not return Bob's memory.
    res = search_memories(store, "what color"; scope="alice", limit=2)
    @test !isempty(res)
    @test all(r -> r.record.scope == "alice", res)
    @test any(r -> occursin("blue", r.record.content), res)
    # Score is a similarity in [0, 1]
    @test all(r -> 0.0 <= r.score <= 1.0, res)

    # get_memories scoped
    alice = get_memories(store; scope="alice")
    @test length(alice) == 2
    @test all(r -> r.scope == "alice", alice)

    # Round-trip preserves role + kind
    @test Set(r.role for r in alice) == Set([:user, :assistant])

    # clear by scope
    clear_memories!(store; scope="bob")
    @test isempty(get_memories(store; scope="bob"))
    @test length(get_memories(store; scope="alice")) == 2
end

@testset "MemPalaceMemoryStore plugs into MemoryContextProvider" begin
    p = Palace(embedder = DeterministicEmbedder(64))
    store = Ext.MemPalaceMemoryStore(palace = p)
    add_memories!(store, [
        MemoryRecord(scope="alice", role=:user,
                      content="Alice is a Julia developer based in Portland."),
        MemoryRecord(scope="alice", role=:assistant,
                      content="Got it — Julia and Portland."),
    ])
    provider = MemoryContextProvider(store = store, max_results = 3)
    @test provider isa BaseContextProvider

    # Smoke: provider exposes the same store and respects max_results.
    @test provider.store === store
    @test provider.max_results == 3
end

@testset "MemPalaceContextProvider — wing/room scoping" begin
    p = Palace(embedder = DeterministicEmbedder(64))

    # Pre-seed two scopes with distinct content.
    mine_text!(p, "Alice prefers dark-mode UIs and lives in Portland.";
                wing="alice", room="profile")
    mine_text!(p, "Bob is the deployment lead and works from Seattle.";
                wing="bob",   room="profile")

    provider = Ext.MemPalaceContextProvider(p;
        wing = nothing,
        room = "profile",
        n_results = 2,
        store = false,  # don't write back during the test
    )
    @test provider isa BaseContextProvider
    @test provider.room == "profile"
    @test provider.store == false

    # Wing comes from session.user_id when not pinned in the constructor.
    sess = AgentSession(user_id = "alice")
    state = Dict{String, Any}()
    ctx = SessionContext()
    push!(ctx.input_messages, Message(:user, "what UI mode does Alice prefer?"))

    AgentFramework.before_run!(provider, nothing, sess, ctx, state)
    @test state["last_query"] == "what UI mode does Alice prefer?"
    @test state["last_result_count"] >= 1

    # Provider injects context under its source_id key in context_messages.
    @test haskey(ctx.context_messages, "mempalace")
    injected = join([m.text for m in ctx.context_messages["mempalace"]], "\n")
    @test occursin("dark", lowercase(injected))
    @test !occursin("Seattle", injected)

    # Switch wing via session metadata
    sess2 = AgentSession()
    sess2.metadata["memory_scope"] = "bob"
    state2 = Dict{String, Any}()
    ctx2 = SessionContext()
    push!(ctx2.input_messages, Message(:user, "where does the deploy lead work?"))
    AgentFramework.before_run!(provider, nothing, sess2, ctx2, state2)
    @test state2["last_result_count"] >= 1
    injected2 = join([m.text for m in ctx2.context_messages["mempalace"]], "\n")
    @test occursin("Seattle", injected2)
end

@testset "MemPalaceContextProvider — store=true round-trips turns" begin
    p = Palace(embedder = DeterministicEmbedder(64))
    provider = Ext.MemPalaceContextProvider(p;
        wing = "session1", room = "chat", n_results = 5, store = true,
    )
    sess = AgentSession()
    ctx = SessionContext()
    push!(ctx.input_messages, Message(:user, "remember that the launch is on Tuesday"))

    state = Dict{String, Any}()
    AgentFramework.after_run!(provider, nothing, sess, ctx, state)
    @test state["stored_count"] >= 1

    # The new drawer should be retrievable.
    hits = MemPalace.search(p, "when is the launch?"; wing="session1", n_results=3)
    @test !isempty(hits)
    @test any(occursin("Tuesday", h.drawer.text) for h in hits)
end
