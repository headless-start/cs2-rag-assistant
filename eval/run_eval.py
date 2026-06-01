"""Offline RAGAS evaluation of the RAG pipeline.

Two phases, so the GPU only holds one set of models at a time:

1. run every question through the pipeline and cache (question, answer,
   contexts, reference) to ``eval/cache/dataset.json``;
2. score that cache with RAGAS using a *local* judge LLM and local embeddings,
   so the whole thing runs with no paid API.

    python -m eval.run_eval                 # generate + score
    python -m eval.run_eval --reuse         # re-score the cached generations
    JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct python -m eval.run_eval --reuse

Every number written to results.md comes from this run.
"""
import argparse
import gc
import json
import os
from datetime import date
from pathlib import Path

import yaml

from src.config import ROOT, settings, resolve_device

QUESTIONS = ROOT / "eval" / "questions.yaml"
CACHE = ROOT / "eval" / "cache" / "dataset.json"
RESULTS = ROOT / "eval" / "results.md"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate_dataset():
    from src.rag import RagPipeline
    items = yaml.safe_load(QUESTIONS.read_text())
    rag = RagPipeline()
    rows = []
    for i, it in enumerate(items, 1):
        res = rag.answer(it["question"], max_new_tokens=256)
        rows.append({
            "question": it["question"],
            "answer": res.answer,
            "contexts": [p.body for p in res.passages],
            "ground_truth": it["ground_truth"],
            "sources": [f"{p.source}>{p.section}" for p in res.passages],
        })
        print(f"  [{i}/{len(items)}] {it['question'][:60]}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, indent=2))
    del rag
    free_gpu()
    return rows


def build_judge():
    """Local judge LLM + embeddings, wrapped for RAGAS."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    name = os.environ.get("JUDGE_MODEL", settings.gen_model)
    device = resolve_device(settings.gen_device)
    kwargs = {"torch_dtype": "auto"}
    if os.environ.get("LLM_4BIT", "1") == "1":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tok,
                    max_new_tokens=512, do_sample=False, return_full_text=False)
    llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))

    emb = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": resolve_device(settings.embed_device)})
    return llm, LangchainEmbeddingsWrapper(emb), name


def score(rows):
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    llm, emb, judge_name = build_judge()
    samples = [SingleTurnSample(
        user_input=r["question"], response=r["answer"],
        retrieved_contexts=r["contexts"], reference=r["ground_truth"])
        for r in rows]
    dataset = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=emb,
        run_config=RunConfig(timeout=240, max_retries=4, max_workers=1),
        show_progress=True,
    )
    return result, judge_name


def write_results(result, judge_name, n):
    df = result.to_pandas()
    means = {m: float(df[m].mean()) for m in METRIC_LABELS if m in df.columns}

    lines = [
        "# RAGAS evaluation",
        "",
        f"- **Date:** {date.today().isoformat()}",
        f"- **Questions:** {n} (see `questions.yaml`)",
        f"- **Generator:** `{settings.gen_model}` (4-bit)",
        f"- **Judge LLM:** `{judge_name}` (4-bit, local)",
        f"- **Judge embeddings:** `{settings.embed_model}`",
        f"- **Retriever:** bge-m3 dense + BM25, RRF fusion, "
        f"bge-reranker-v2-m3 rerank (top {settings.rerank_top_n})",
        "",
        "## Scores",
        "",
        "| Metric | Score |",
        "|--------|------:|",
    ]
    for m, label in METRIC_LABELS.items():
        if m in means:
            lines.append(f"| {label} | {means[m]:.3f} |")
    lines += [
        "",
        "Faithfulness and answer relevancy score the generated answer against "
        "the retrieved passages; context precision and recall score the "
        "retriever against the reference answers. Scores are produced by a "
        "local open judge model, so treat them as a directional, reproducible "
        "signal rather than an absolute ground truth.",
        "",
        "## Per-question scores",
        "",
        "| # | Question | " + " | ".join(METRIC_LABELS[m] for m in METRIC_LABELS if m in df.columns) + " |",
        "|---|----------|" + "|".join(["------:"] * len([m for m in METRIC_LABELS if m in df.columns])) + "|",
    ]
    qcol = "user_input" if "user_input" in df.columns else "question"
    for i, row in df.iterrows():
        cells = [f"{row[m]:.2f}" if m in df.columns and row[m] == row[m] else "—"
                 for m in METRIC_LABELS if m in df.columns]
        q = str(row[qcol])
        q = (q[:57] + "…") if len(q) > 58 else q
        lines.append(f"| {i + 1} | {q} | " + " | ".join(cells) + " |")
    lines.append("")
    RESULTS.write_text("\n".join(lines))
    print(f"\nwrote {RESULTS}")
    print({k: round(v, 3) for k, v in means.items()})
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="re-score cached generations instead of regenerating")
    args = ap.parse_args()

    if args.reuse and CACHE.exists():
        rows = json.loads(CACHE.read_text())
        print(f"reusing {len(rows)} cached generations")
        free_gpu()
    else:
        print("generating answers…")
        rows = generate_dataset()

    print("scoring with RAGAS…")
    result, judge_name = score(rows)
    write_results(result, judge_name, len(rows))


if __name__ == "__main__":
    main()
