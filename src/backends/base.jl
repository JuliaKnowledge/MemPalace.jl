# Backend abstraction. Mirrors `mempalace/backends/base.py` but adapted to
# Julia idioms — drawers and closets are typed Julia objects rather than
# raw dicts, and embeddings are stored alongside each record so the
# backend owns vector retrieval.

"""
    AbstractPalaceBackend

The pluggable storage contract for MemPalace. Concrete subtypes must
implement *all* of:

  * [`add_drawer!(b, drawer, embedding)`](@ref add_drawer!)
  * [`upsert_drawer!(b, drawer, embedding)`](@ref upsert_drawer!)
  * [`get_drawer(b, id)`](@ref get_drawer)
  * [`query_drawers(b, query_embedding, query_text; ...)`](@ref query_drawers)
  * [`add_closet!(b, closet, embedding)`](@ref add_closet!)
  * [`upsert_closet!(b, closet, embedding)`](@ref upsert_closet!)
  * [`query_closets(b, query_embedding; ...)`](@ref query_closets)
  * [`delete_drawers!(b; source_file=nothing, ids=nothing)`](@ref delete_drawers!)
  * [`count_drawers(b)`](@ref count_drawers)
  * [`count_closets(b)`](@ref count_closets)
  * [`clear!(b)`](@ref clear!)

The contract is intentionally narrower than the upstream Python
`BaseCollection` (no `update`, `**kwargs`-based `query`/`get`/`delete`)
because Julia multiple-dispatch encodes the variants in distinct method
signatures.
"""
abstract type AbstractPalaceBackend end

# Generic API hooks. Concrete backends specialise each method.
function add_drawer! end
function upsert_drawer! end
function get_drawer end
function query_drawers end
function add_closet! end
function upsert_closet! end
function query_closets end
function delete_drawers! end
function count_drawers end
function count_closets end
function clear! end
