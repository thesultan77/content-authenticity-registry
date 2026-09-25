# v0.2.16
# {
#   "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
#   ]
# }
import numpy as np
from genlayer import *
import genlayer_embeddings as gle
import datetime
import typing
import json
from dataclasses import dataclass


@allow_storage
@dataclass
class Entry:
    entry_id: str
    author: Address
    title: str
    content_excerpt: str


class ContentAuthenticityRegistry(gl.Contract):
    entries: gle.VecDB[np.float32, typing.Literal[384], Entry]
    authors_map: TreeMap[str, Address]
    titles_map: TreeMap[str, str]
    content_map: TreeMap[str, str]
    registered_at_map: TreeMap[str, str]
    flag_status_map: TreeMap[str, str]
    flag_reason_map: TreeMap[str, str]
    flagged_against_map: TreeMap[str, str]
    entry_counter: u256

    def __init__(self):
        self.entry_counter = u256(0)

    def get_embedding_generator(self):
        return gle.SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(
        self, txt: str
    ) -> np.ndarray[tuple[typing.Literal[384]], np.dtypes.Float32DType]:
        return self.get_embedding_generator()(txt)

    @gl.public.write
    def register_content(self, title: str, content: str) -> str:
        entry_id = f"E{self.entry_counter}"
        self.entry_counter += u256(1)

        excerpt = content[:2000]
        emb = self.get_embedding(content)

        self.entries.insert(emb, Entry(
            entry_id=entry_id,
            author=gl.message.sender_address,
            title=title,
            content_excerpt=excerpt,
        ))

        self.authors_map[entry_id] = gl.message.sender_address
        self.titles_map[entry_id] = title
        self.content_map[entry_id] = excerpt
        self.registered_at_map[entry_id] = str(datetime.datetime.now())
        self.flag_status_map[entry_id] = "CLEAR"
        self.flag_reason_map[entry_id] = ""
        self.flagged_against_map[entry_id] = ""

        return entry_id

    @gl.public.view
    def check_similar(self, content: str, top_n: int) -> list[dict]:
        emb = self.get_embedding(content)
        out = []
        for r in self.entries.knn(emb, top_n):
            out.append({
                "similarity": str(r.distance),
                "entry_id": r.value.entry_id,
                "title": r.value.title,
                "author": str(r.value.author),
            })
        return out

    @gl.public.write
    def flag_entry(self, entry_id: str) -> str:
        assert self.flag_status_map[entry_id] == "CLEAR", "Entry already reviewed"

        entry_content = self.content_map[entry_id]

        matches = list(self.entries.knn(self.get_embedding(entry_content), 5))
        candidate = None
        for m in matches:
            if m.value.entry_id != entry_id:
                candidate = m
                break

        if candidate is None:
            self.flag_status_map[entry_id] = "CLEAR"
            self.flag_reason_map[entry_id] = "No comparable prior entry found."
            return "CLEAR - no comparable prior entry found"

        excerpt_a = candidate.value.content_excerpt
        excerpt_b = entry_content

        def judge() -> str:
            prompt = f"""You are checking two pieces of content for substantive originality overlap.

Prior registered content:
\"\"\"{excerpt_a}\"\"\"

New content being reviewed:
\"\"\"{excerpt_b}\"\"\"

Is the new content substantially the same underlying idea, argument, or creative work as the prior one (reused/derivative/paraphrased), or is any similarity just coincidental topical overlap (both legitimately original)?

Respond with strict JSON only, no markdown, no extra text:
{{"verdict": "DUPLICATE", "reasoning": "<one sentence>"}}
or
{{"verdict": "ORIGINAL", "reasoning": "<one sentence>"}}
"""
            return gl.nondet.exec_prompt(prompt)

        raw = gl.eq_principle.prompt_comparative(
            judge,
            "The verdict field must match exactly (DUPLICATE or ORIGINAL). Reasoning wording may vary but must express the same underlying judgment.",
        )
        parsed = json.loads(raw)
        verdict = parsed["verdict"]
        reasoning = parsed["reasoning"]

        if verdict == "DUPLICATE":
            self.flag_status_map[entry_id] = "FLAGGED"
            self.flag_reason_map[entry_id] = reasoning
            self.flagged_against_map[entry_id] = candidate.value.entry_id
            return f"FLAGGED - {reasoning} (against {candidate.value.entry_id})"
        else:
            self.flag_status_map[entry_id] = "CLEAR"
            self.flag_reason_map[entry_id] = reasoning
            return f"CLEAR - {reasoning}"

    @gl.public.view
    def get_entry_status(self, entry_id: str) -> dict:
        return {
            "entry_id": entry_id,
            "author": str(self.authors_map[entry_id]),
            "title": self.titles_map[entry_id],
            "registered_at": self.registered_at_map[entry_id],
            "status": self.flag_status_map[entry_id],
            "reason": self.flag_reason_map[entry_id],
            "flagged_against": self.flagged_against_map[entry_id],
        }

    @gl.public.view
    def get_entry_count(self) -> int:
        return int(self.entry_counter)
