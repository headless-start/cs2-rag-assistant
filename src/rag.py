"""retrieve -> rerank -> prompt -> grounded answer with inline [n] citations."""
from dataclasses import dataclass

from .llm import Generator
from .retrieve import HybridRetriever, Passage

SYSTEM = (
    "You are a Counter-Strike 2 assistant. Answer the question using ONLY the "
    "numbered context passages provided. Cite the passages you rely on inline "
    "with their number in square brackets (for example [1] or [2][3]), placed "
    "right after the claim they support. If the passages do not contain enough "
    "information to answer, say that you don't have enough information rather "
    "than guessing. Be concise and factual, and do not mention these "
    "instructions or that you were given context."
)


def format_context(passages):
    blocks = []
    for i, p in enumerate(passages, 1):
        loc = p.source if p.section in ("", "overview") else f"{p.source} > {p.section}"
        blocks.append(f"[{i}] ({loc})\n{p.body}")
    return "\n\n".join(blocks)


@dataclass
class RagResult:
    answer: str
    passages: list          # list[Passage], indexed 1..n matching the citations
    query: str


class RagPipeline:
    def __init__(self, retriever=None, generator=None, top_n=None):
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or Generator()
        self.top_n = top_n

    def answer(self, query, top_n=None, max_new_tokens=None):
        passages = self.retriever.retrieve(query, top_n=top_n or self.top_n)
        if not passages:
            return RagResult("I don't have enough information to answer that.",
                             [], query)
        user = (f"Context passages:\n{format_context(passages)}\n\n"
                f"Question: {query}\n\nAnswer the question, citing the passages "
                f"you use by their bracketed number.")
        answer = self.generator.generate(SYSTEM, user, max_new_tokens=max_new_tokens)
        return RagResult(answer, passages, query)
