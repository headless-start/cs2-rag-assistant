"""Load the markdown corpus and split it into citeable chunks.

Chunking is markdown-aware: we split on ``##`` section headings first so a
chunk never straddles two topics, then sub-split any section that is longer
than the target size, with a little overlap. Every chunk keeps its source file
and section heading so the generator can cite it.
"""
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings


@dataclass
class Chunk:
    id: str
    text: str          # what we embed / show, includes a "title - section" header
    body: str          # the raw passage without the header line
    source: str        # file name, e.g. map-mirage.md
    title: str         # document title (first H1)
    section: str       # nearest H2, or "overview"

    def as_payload(self):
        return asdict(self)


# rough token estimate so ingest does not have to load a tokenizer; CS2 prose
# sits around 1.3 tokens per whitespace word.
def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def _split_sections(md):
    """Return [(section_title, body)] splitting on H2 headings."""
    lines = md.splitlines()
    title = "untitled"
    sections = []
    cur_title = "overview"
    cur = []
    for line in lines:
        h1 = re.match(r"^#\s+(.*)", line)
        h2 = re.match(r"^##\s+(.*)", line)
        if h1:
            title = h1.group(1).strip()
            continue
        if h2:
            if cur and any(l.strip() for l in cur):
                sections.append((cur_title, "\n".join(cur).strip()))
            cur_title = h2.group(1).strip()
            cur = []
            continue
        cur.append(line)
    if cur and any(l.strip() for l in cur):
        sections.append((cur_title, "\n".join(cur).strip()))
    return title, sections


def _split_long(body, max_tokens, overlap_tokens):
    """Split an over-long section on paragraph boundaries with word overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, cur = [], []
    cur_tok = 0
    for p in paras:
        pt = estimate_tokens(p)
        if cur and cur_tok + pt > max_tokens:
            chunks.append("\n\n".join(cur))
            # carry the tail of the previous chunk as overlap
            tail = " ".join("\n\n".join(cur).split()[-overlap_tokens:])
            cur, cur_tok = ([tail] if tail else []), estimate_tokens(tail)
        cur.append(p)
        cur_tok += pt
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _emit(out, path, title, group, idx):
    """Build one Chunk from a list of (section_title, body) pieces."""
    head_section = group[0][0]
    body = "\n\n".join(
        (f"## {s}\n{b}" if s != "overview" else b) for s, b in group
    ).strip()
    label = title if head_section == "overview" else f"{title} — {head_section}"
    cid = f"{path.stem}::{idx}"
    out.append(Chunk(
        id=cid,
        text=f"{label}\n\n{body}",
        body=body,
        source=path.name,
        title=title,
        section=head_section,
    ))


def chunk_file(path: Path, max_tokens, overlap_tokens):
    """Pack a file's sections into ~max_tokens chunks, splitting a section only
    if it alone exceeds the budget."""
    title, sections = _split_sections(path.read_text(encoding="utf-8"))
    out, group, group_tok, idx = [], [], 0, 0
    for s_title, body in sections:
        if estimate_tokens(body) > max_tokens:
            if group:
                _emit(out, path, title, group, idx); idx += 1
                group, group_tok = [], 0
            for piece in _split_long(body, max_tokens, overlap_tokens):
                _emit(out, path, title, [(s_title, piece)], idx); idx += 1
            continue
        if group and group_tok + estimate_tokens(body) > max_tokens:
            _emit(out, path, title, group, idx); idx += 1
            group, group_tok = [], 0
        group.append((s_title, body))
        group_tok += estimate_tokens(body)
    if group:
        _emit(out, path, title, group, idx)
    return out


def load_chunks(corpus_dir=None, max_tokens=None, overlap_tokens=None):
    corpus_dir = Path(corpus_dir or settings.corpus_dir)
    max_tokens = max_tokens or settings.chunk_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_file(path, max_tokens, overlap_tokens))
    return chunks


if __name__ == "__main__":
    cs = load_chunks()
    toks = [estimate_tokens(c.body) for c in cs]
    print(f"{len(cs)} chunks from {len(set(c.source for c in cs))} files")
    print(f"tokens: min {min(toks)}  max {max(toks)}  mean {sum(toks)//len(toks)}")
