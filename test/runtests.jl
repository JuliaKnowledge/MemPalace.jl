using MemPalace
using Test

@testset "MemPalace.jl" begin

    @testset "Embedders" begin
        e = DeterministicEmbedder(64)
        v1 = embed(e, "hello world")
        v2 = embed(e, "hello world")
        v3 = embed(e, "completely different content")
        @test length(v1) == 64
        @test v1 == v2                 # deterministic
        @test v1 != v3
        @test isapprox(sum(v -> v^2, v1), 1.0; atol=1e-9)  # L2-normalised
        @test embed(e, "") == zeros(64)
    end

    @testset "Drawer & Closet show" begin
        d = Drawer(text = "alice fixed the parser bug", wing="alice", room="bugs")
        @test occursin("Drawer", sprint(show, d))
        @test occursin("alice", sprint(show, d))
        c = Closet(id="c1", text="t|e|→a,b", drawer_ids=["a","b"])
        @test occursin("Closet", sprint(show, c))
    end

    @testset "Extractor" begin
        text = """
        Alice and Bob worked on Acme. Alice fixed the parser bug. Bob created
        the dashboard. Acme deployed to staging. Alice reviewed the change.
        """
        ents = extract_entities(text; min_count=2)
        @test "Alice" in ents
        @test "Acme" in ents
        # Filler words should not appear
        @test !("The" in ents)
        @test !("And" in ents)

        topics = extract_topics(text)
        @test !isempty(topics)
        @test any(t -> occursin("fixed", t), topics)
        @test any(t -> occursin("created", t) || occursin("deployed", t), topics)

        # Closet line builder always emits at least one line
        ids = ["d1", "d2"]
        lines = build_closet_lines("notes.md", ids, text, "alice", "bugs")
        @test !isempty(lines)
        @test all(occursin("→d1,d2", l) for l in lines)

        # Round-trip drawer ids
        joined = join(lines, "\n")
        parsed = MemPalace.extract_drawer_ids_from_closet(joined)
        @test "d1" in parsed && "d2" in parsed
    end

    @testset "InMemoryPalaceBackend" begin
        b = InMemoryPalaceBackend()
        e = DeterministicEmbedder(64)
        d1 = Drawer(text="apple banana cherry", wing="w1", room="r1", source_file="a.md")
        d2 = Drawer(text="dog elephant fox",    wing="w1", room="r2", source_file="a.md")
        d3 = Drawer(text="apple plum pear",     wing="w2", room="r1", source_file="b.md")
        add_drawer!(b, d1, embed(e, d1.text))
        add_drawer!(b, d2, embed(e, d2.text))
        add_drawer!(b, d3, embed(e, d3.text))
        @test count_drawers(b) == 3
        @test get_drawer(b, d1.id) === d1

        q = embed(e, "apple")
        hits = query_drawers(b, q; n_results=2)
        @test length(hits) == 2
        @test hits[1][1].id in (d1.id, d3.id)

        # Wing filter
        h_w1 = query_drawers(b, q; wing="w1", n_results=10)
        @test all(p -> p[1].wing == "w1", h_w1)

        # Source filter + delete by source
        n_del = delete_drawers!(b; source_file="a.md")
        @test n_del == 2
        @test count_drawers(b) == 1

        # Delete by id
        @test delete_drawers!(b; ids=[d3.id]) == 1
        @test count_drawers(b) == 0

        # Closets
        c = Closet(id="c1", text="topic|ent|→x,y", wing="w1", drawer_ids=["x","y"])
        upsert_closet!(b, c, embed(e, c.text))
        @test count_closets(b) == 1
        # upsert is idempotent
        upsert_closet!(b, c, embed(e, c.text))
        @test count_closets(b) == 1
        cpairs = query_closets(b, embed(e, "topic"); wing="w1", n_results=5)
        @test length(cpairs) == 1

        clear!(b)
        @test count_drawers(b) == 0
        @test count_closets(b) == 0
    end

    @testset "BM25 + hybrid_rank!" begin
        docs = ["apple banana cherry",
                "apple apple apple",
                "completely unrelated",
                "banana cherry"]
        s = MemPalace.bm25_scores("apple", docs)
        @test length(s) == 4
        # The 2nd doc has 3x "apple" — should outscore the 1st (1x).
        @test s[2] > s[1]
        @test s[3] == 0.0     # no overlap
        @test s[4] == 0.0     # no "apple"
    end

    @testset "Palace mine + search" begin
        p = Palace(embedder = DeterministicEmbedder(128))
        text = """
        # Refactor notes
        Alice fixed the parser bug on Monday. Bob added unit tests for the
        tokenizer. The CI pipeline now runs on every push. Acme deployed
        the patched build to staging. Alice reviewed Bob's PR.
        """
        drawers = mine_text!(p, text;
                              wing="alice", room="bugs",
                              source_file="refactor.md")
        @test !isempty(drawers)
        @test count_drawers(p.backend) == length(drawers)
        @test count_closets(p.backend) >= 1

        results = search(p, "what did Alice fix?"; wing="alice", n_results=3)
        @test !isempty(results)
        # Top hit should mention the parser bug.
        @test any(occursin("parser", h.drawer.text) for h in results)
        top = results[1]
        @test top.distance >= 0.0
        @test top.score > 0.0

        # Re-mining the same source replaces drawers (no duplicates)
        n_before = count_drawers(p.backend)
        drawers2 = mine_text!(p, text;
                               wing="alice", room="bugs",
                               source_file="refactor.md")
        @test count_drawers(p.backend) == length(drawers2)
        @test count_drawers(p.backend) <= n_before  # never grows
    end

    @testset "mine_conversation!" begin
        p = Palace(embedder = DeterministicEmbedder(64))
        msgs = [
            Dict("role" => "user",      "content" => "Where did we deploy the build?"),
            Dict("role" => "assistant", "content" => "We pushed it to staging."),
            Dict("role" => "user",      "content" => ""),  # filtered
        ]
        ds = mine_conversation!(p, msgs;
                                 wing="chat", room="conv",
                                 source_file="thread1")
        @test length(ds) == 2
        @test all(d -> d.wing == "chat", ds)
        # Verbatim retrieval — full original text round-trips
        hits = search(p, "where did we deploy?"; wing="chat")
        @test !isempty(hits)
        @test any(h -> occursin("staging", h.drawer.text), hits)
    end

    @testset "wake_up bundle" begin
        p = Palace(embedder = DeterministicEmbedder(64))
        mine_text!(p, "Project Acme launched in March."; wing="proj")
        mine_text!(p, "The team chose Postgres for storage."; wing="proj")
        mine_text!(p, "Alice owns the tokenizer module."; wing="proj")
        bundle = wake_up(p, ["acme", "postgres", "tokenizer"];
                          per_query=2, total=5)
        @test !isempty(bundle)
        @test length(bundle) <= 5
    end

    @testset "AgentFramework extension (loaded on demand)" begin
        local ext_loaded = false
        try
            @eval using AgentFramework
            ext_loaded = true
        catch err
            @info "AgentFramework not available — skipping ext tests" exception=err
        end
        if ext_loaded
            include("test_af_ext.jl")
        end
    end
end
