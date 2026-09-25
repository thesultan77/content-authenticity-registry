# ContentAuthenticityRegistry

**On-Chain Content Originality Registry with AI-Judged Duplicate Detection**

A registry contract where writers and creators register their work, and
anyone can flag a new entry for an AI-judged originality check — distinguishing
genuine plagiarism/derivative reuse from harmless coincidental topical overlap.

---

## What it does

1. Anyone can **register content** (a title + body text). The content is
   embedded into a vector and stored in a `VecDB`, alongside the author's
   address and a timestamp.
2. Anyone can **check similarity** of a new piece of text against everything
   already registered, before committing to anything on-chain — a live
   semantic search over the registry.
3. Anyone can **flag an entry** for a formal originality review. The contract
   finds the closest prior registered entry via vector similarity, then asks
   an LLM to judge whether the new content is a **substantial reuse/paraphrase**
   of the prior work (`DUPLICATE`) or just **coincidental topical overlap**
   between two legitimately original pieces (`ORIGINAL`). GenLayer's validator
   consensus (`gl.eq_principle.prompt_comparative`) ensures multiple
   independent LLM runs agree on the categorical verdict before it's
   accepted on-chain.
4. The verdict, reasoning, and (if flagged) which prior entry it was flagged
   against are all recorded permanently and queryable per entry.

## Why this is useful

Plagiarism and content-reuse disputes are usually adjudicated by a human
editor eyeballing two pieces of text — subjective, slow, and with no
permanent public record of the reasoning. This contract makes the
originality check itself a transparent, consensus-verified, on-chain event:
anyone can see exactly what was compared, what the AI judge concluded, and
why — with the semantic pre-filtering (`VecDB` similarity search) making
sure the LLM is only ever asked to compare genuinely close candidates rather
than every entry in the registry.

## Architecture

| Concern | Mechanism |
|---|---|
| Content storage & search | `VecDB[float32, 384, Entry]` — vector search over registered work |
| Per-entry metadata | Parallel `TreeMap[str, ...]` fields (author, title, full content, timestamp, flag status, reasoning, flagged-against) |
| Embeddings | `SentenceTransformer("all-MiniLM-L6-v2")` via `genlayer_embeddings`, called as `model(text)` |
| AI judgment | `gl.nondet.exec_prompt` inside a closure, reconciled across validators via `gl.eq_principle.prompt_comparative` |

Entry metadata is stored as parallel primitive `TreeMap`s (one map per
field) rather than a single `TreeMap` of a `@allow_storage` dataclass —
constructing a fresh dataclass via keyword arguments and assigning it
directly into a `TreeMap` triggers a GenVM storage-serialization error in
the current Studio build. The same dataclass pattern works fine as a
`VecDB` value type, so the `VecDB` (`Entry`) uses a dataclass while the
`TreeMap`s use primitives only.

## Contract methods

### Write

- **`register_content(title: str, content: str) -> str`**
  Registers a new piece of content under the caller's address. Returns the
  new `entry_id` (e.g. `"E0"`).
- **`flag_entry(entry_id: str) -> str`**
  Triggers the AI originality review for an entry that hasn't been reviewed
  yet: finds the closest prior entry by vector similarity, prompts the LLM
  for a `DUPLICATE`/`ORIGINAL` verdict with reasoning, reaches validator
  consensus, and records the result. Returns `"<VERDICT> - <reasoning>"`.

### View

- **`check_similar(content: str, top_n: int) -> list[dict]`**
  Preview which registered entries are semantically closest to a given piece
  of text, without registering or flagging anything.
- **`get_entry_status(entry_id: str) -> dict`**
  Full current state of an entry (author, title, timestamp, flag status,
  reasoning, which entry it was flagged against if any).
- **`get_entry_count() -> int`**

## Tested end-to-end in GenLayer Studio

- Registered three entries: two near-paraphrases of the same renewable
  energy piece, and one unrelated sourdough-bread guide.
- `check_similar` correctly surfaced the closest related entry with a low
  distance score for a semantically matching query.
- `flag_entry` on the paraphrased entry correctly returned **`FLAGGED`**,
  with the LLM's reasoning specifically identifying the shared structure,
  claims, and sequence of ideas between the two texts.
- `get_entry_status` confirmed the `FLAGGED` verdict, reasoning, and
  `flagged_against` reference were all persisted correctly.
- `flag_entry` on the unrelated sourdough entry correctly returned
  **`CLEAR`/`ORIGINAL`**, with reasoning correctly noting the two pieces
  cover entirely unrelated topics.
- All test transactions reached `FINALIZED` / `SUCCESS` with supermajority
  validator agreement across multiple LLM policies.

## Deployment

```
# v0.2.16
# { "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
# ] }
```

Constructor takes no arguments. Deploy directly in GenLayer Studio or via
`genlayer-js` against Studionet / Bradbury testnet.

## Notes on GenVM quirks discovered while building this

- `gle.SentenceTransformer("all-MiniLM-L6-v2")` returns a **callable**, not
  an object with `.encode()`. Call it directly: `model(text)`, not
  `model.encode(text)`.
- Constructing a new `@allow_storage` dataclass instance via keyword
  arguments and assigning it directly into a `TreeMap` value slot can throw
  a GenVM storage serialization error; `VecDB` as the dataclass's container
  works fine, and parallel primitive `TreeMap`s are a reliable workaround
  when structured per-key records are needed outside a `VecDB`.
- When comparing content for similarity/originality, make sure the
  *original full text* is what gets embedded and passed to the LLM for
  judgment — not a short label like a title. Comparing a title against a
  full passage produces unreliable, misleadingly "ORIGINAL" verdicts simply
  because a six-word title never looks substantially similar to a paragraph.

## License

MIT
