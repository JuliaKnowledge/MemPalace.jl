# Backend abstraction. Mirrors `mempalace/backends/base.py` but adapted to
# Julia idioms — drawers and closets are typed Julia objects rather than
# raw dicts, and embeddings are stored alongside each record so the
# backend owns vector retrieval.

"""
    AbstractPalaceBackend

The pluggable storage contract for MemPalace. Concrete subtypes must
implement *all* of:

  * `add_drawer!(b, drawer, embedding)`
  * `upsert_drawer!(b, drawer, embedding)`
  * `get_drawer(b, id)`
  * `query_drawers(b, query_embedding, query_text; ...)`
  * `add_closet!(b, closet, embedding)`
  * `upsert_closet!(b, closet, embedding)`
  * `query_closets(b, query_embedding; ...)`
  * `delete_drawers!(b; source_file=nothing, ids=nothing)`
  * `count_drawers(b)`
  * `count_closets(b)`
  * `clear!(b)`

The contract is intentionally narrower than the upstream Python
`BaseCollection` (no `update`, `**kwargs`-based `query`/`get`/`delete`)
because Julia multiple-dispatch encodes the variants in distinct method
signatures.
"""
abstract type AbstractPalaceBackend end

"Insert a `Drawer` together with its precomputed embedding into the backend. Errors if the drawer id already exists."
function add_drawer! end
"Insert or update a `Drawer` (with embedding) by id."
function upsert_drawer! end
function get_drawer end
function query_drawers end
"Insert a `Closet` together with its precomputed embedding."
function add_closet! end
"Insert or update a `Closet` (with embedding) by id."
function upsert_closet! end
function query_closets end
function delete_drawers! end
function count_drawers end
function count_closets end
function clear! end
