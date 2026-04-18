# API Reference

## Core Types

```@docs
MemPalace.Drawer
MemPalace.Closet
MemPalace.Palace
MemPalace.SearchHit
MemPalace.SearchResults
```

## Backends

```@docs
MemPalace.AbstractPalaceBackend
MemPalace.InMemoryPalaceBackend
MemPalace.add_drawer!
MemPalace.upsert_drawer!
MemPalace.add_closet!
MemPalace.upsert_closet!
```

## Embedders

```@docs
MemPalace.AbstractMemPalaceEmbedder
MemPalace.DeterministicEmbedder
MemPalace.OllamaEmbedder
MemPalace.embed
```

## Extraction & Ranking

```@docs
MemPalace.build_closet_lines
MemPalace.extract_entities
MemPalace.extract_topics
MemPalace.bm25_scores
MemPalace.hybrid_rank!
```

## Mining and Retrieval

```@docs
MemPalace.mine_text!
MemPalace.mine_conversation!
MemPalace.search
MemPalace.wake_up
```

## Backend Queries (in-memory)

```@docs
MemPalace.query_drawers
MemPalace.query_closets
MemPalace.delete_drawers!
```
