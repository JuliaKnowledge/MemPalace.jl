# MemPalace.jl quickstart vignette
#
# Narrative: a small dev team keeps a rolling CHANGELOG. We mine it into
# a palace scoped by author (wing) and topic (room), then retrieve
# verbatim snippets across arbitrary queries. Finally we run `wake_up`
# on a bundle of probes to warm a new session with the most salient
# facts from each wing.
#
# What this demonstrates
#   - `Palace` with the default in-memory backend + deterministic embedder
#   - `mine_text!` with `source_file` (idempotent replace on re-mine)
#   - `search` with wing / room filters + closet boost
#   - `wake_up` for session warm-up
#
# Run:  julia --project=. examples/quickstart.jl

using MemPalace
using Test

# chunk_chars=250 so each bullet-section becomes its own drawer; makes the
# retrieval stories more interesting without needing a huge corpus.
p = Palace(chunk_chars = 250, chunk_overlap = 30)
@info "Created palace" p

changelog_alice = """
# 2025-02-10  Alice
- Fixed the parser bug that crashed on empty input.
- Added fuzzer coverage for the lexer.
- Reviewed Bob's tokenizer PR.

# 2025-02-12  Alice
- Shipped v0.4.1 to staging.
- Updated the docs site with the new parser section.
"""

changelog_bob = """
# 2025-02-10  Bob
- Replaced the hand-rolled tokenizer with a regex-driven state machine.
- Wrote 42 unit tests; 3 of them caught regressions in the old code.

# 2025-02-12  Bob
- Paired with Carol on the Postgres migration runbook.
- Cut the staging deploy time from 14m to 4m.
"""

mine_text!(p, changelog_alice;
           wing="alice", room="changelog",
           source_file="changelog-alice.md")
mine_text!(p, changelog_bob;
           wing="bob",   room="changelog",
           source_file="changelog-bob.md")

@info "Mined" drawers=count_drawers(p.backend) closets=count_closets(p.backend)
@test count_drawers(p.backend) >= 2
@test count_closets(p.backend) >= 2

# --- basic retrieval (no filter) ------------------------------------------
hits = search(p, "who fixed the parser bug?"; n_results=3)
@info "unfiltered" top=first(hits.hits).drawer.text
@test any(occursin("parser", h.drawer.text) for h in hits)

# --- wing filter narrows correctly ----------------------------------------
alice_hits = search(p, "what shipped last week?"; wing="alice", n_results=5)
bob_hits   = search(p, "what shipped last week?"; wing="bob",   n_results=5)
@test all(h -> h.drawer.wing == "alice", alice_hits)
@test all(h -> h.drawer.wing == "bob",   bob_hits)

# --- idempotent re-mine (source_file deletes prior drawers first) ---------
new_alice = changelog_alice * "\n# 2025-02-13  Alice\n- Rolled v0.4.1 to prod.\n"
mine_text!(p, new_alice; wing="alice", room="changelog",
           source_file="changelog-alice.md")
prod_hits = search(p, "when did we reach prod?"; wing="alice", n_results=3)
@test any(occursin("prod", h.drawer.text) for h in prod_hits)

# No duplicates from the overlapping mine.
all_alice = search(p, "parser"; wing="alice", n_results=50)
ids = [h.drawer.id for h in all_alice]
@test length(ids) == length(unique(ids))

# --- wake_up: warm a new session with a probe bundle ----------------------
probes = [
    "recent bug fixes",
    "deployment notes",
    "code reviews",
]
warm = wake_up(p, probes; per_query=2, total=6)
@info "wake_up hits" count=length(warm.hits)
@test length(warm.hits) >= 2
for h in warm
    println("  [", h.drawer.wing, "/", h.drawer.room, "] ",
            first(h.drawer.text, 80))
end

println("\nPASS — MemPalace.jl quickstart vignette")
