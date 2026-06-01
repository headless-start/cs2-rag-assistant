# RAGAS evaluation

- **Date:** 2026-06-01
- **Questions:** 38 (see `questions.yaml`)
- **Generator:** `Qwen/Qwen2.5-3B-Instruct` (4-bit)
- **Judge LLM:** `Qwen/Qwen2.5-3B-Instruct` (4-bit, local)
- **Judge embeddings:** `BAAI/bge-m3`
- **Retriever:** bge-m3 dense + BM25, RRF fusion, bge-reranker-v2-m3 rerank (top 5)

## Scores

| Metric | Score | Questions scored |
|--------|------:|-----------------:|
| Faithfulness | 0.746 | 19 / 38 |
| Answer relevancy | 0.866 | 35 / 38 |
| Context precision | 0.934 | 15 / 38 |
| Context recall | 0.816 | 38 / 38 |

Faithfulness and answer relevancy score the generated answer against the
retrieved passages; context precision and recall score the retriever against the
reference answers.

The judge here is the same small local model used for generation
(`Qwen2.5-3B-Instruct`, 4-bit), chosen so the whole evaluation runs offline with
no paid API. A model that size does not always return RAGAS's stricter
structured-output prompts in a parseable form, so faithfulness and context
precision are averaged only over the questions that scored cleanly (the counts
above); context recall and answer relevancy parse reliably across the full set.
Re-running the same cached generations through a stronger judge —
`python -m eval.run_eval --reuse` with `LLM_PROVIDER=openai` or a larger local
model — closes that gap. The numbers here are a reproducible local-only
baseline, not an absolute ceiling.

## Per-question scores

| # | Question | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|----------|------:|------:|------:|------:|
| 1 | How much does the AK-47 cost and what makes it so strong? | 1.00 | 0.92 | — | 1.00 |
| 2 | What is the price of the AWP and why is its kill reward o… | 1.00 | 0.83 | — | 1.00 |
| 3 | What is the difference between the M4A4 and the M4A1-S? | — | 1.00 | — | 1.00 |
| 4 | Which pistol is the standard eco-round buy and what does … | — | 0.79 | 0.81 | 1.00 |
| 5 | What is the SSG 08 and when is it used? | — | 0.95 | — | 1.00 |
| 6 | How much do SMG kills reward, and what is the exception? | 0.50 | 0.87 | — | 1.00 |
| 7 | How much does a shotgun kill pay? | 1.00 | 0.96 | — | 1.00 |
| 8 | What does the Desert Eagle cost and what is its trade-off? | 0.25 | 0.94 | 1.00 | 1.00 |
| 9 | What is the loss bonus ladder in CS2? | 0.00 | 1.00 | 1.00 | 1.00 |
| 10 | How much money do players start each half with, and what … | — | 0.87 | 1.00 | 1.00 |
| 11 | What bonus do the terrorists get for planting the bomb on… | — | 0.61 | — | 1.00 |
| 12 | How much money do you get for winning a round by eliminat… | 0.00 | 0.86 | — | 1.00 |
| 13 | What is the difference between a force buy and an eco? | — | 0.73 | — | 0.00 |
| 14 | Why is saving weapons on a lost round important for the e… | — | 0.97 | 1.00 | 1.00 |
| 15 | How much does Kevlar with a helmet cost and why is it wor… | 0.25 | 0.97 | — | 1.00 |
| 16 | What does a defuse kit cost and what does it do? | 0.67 | 0.98 | — | 1.00 |
| 17 | What is the price difference between the molotov and the … | 1.00 | 0.65 | 1.00 | 1.00 |
| 18 | How much does a flashbang cost and how can you reduce its… | 0.50 | 0.91 | — | 0.00 |
| 19 | How long does a smoke last in CS2? | 1.00 | — | 0.75 | 1.00 |
| 20 | What happens when an HE grenade detonates next to a smoke… | 1.00 | 1.00 | 1.00 | 1.00 |
| 21 | What does the decoy grenade do and how much does it cost? | 1.00 | 0.91 | — | 1.00 |
| 22 | What is the Zeus x27 and how much does it cost? | — | — | — | 1.00 |
| 23 | What is "banana" on Inferno? | 1.00 | 0.36 | — | 1.00 |
| 24 | What is the key to controlling Mirage? | 1.00 | — | — | 0.00 |
| 25 | Why is Nuke considered a CT-sided map? | — | 0.91 | 0.76 | 1.00 |
| 26 | What is unique about Vertigo's environment? | — | 0.81 | 1.00 | 0.00 |
| 27 | What is the defining feature of the map Anubis? | — | 1.00 | — | 1.00 |
| 28 | Why does Dust II play differently from utility-heavy maps? | — | 0.86 | 1.00 | 0.00 |
| 29 | How long is the round timer and how long is the bomb time… | 1.00 | 0.92 | — | 1.00 |
| 30 | How long does it take to plant and to defuse the bomb? | 1.00 | 0.99 | — | 1.00 |
| 31 | What is the MR12 format in CS2? | 1.00 | 1.00 | — | 1.00 |
| 32 | What happens if the score reaches 12-12? | — | 0.85 | 1.00 | 1.00 |
| 33 | What rating system does Premier mode use? | — | 0.69 | — | 1.00 |
| 34 | What does an entry fragger do? | — | 0.96 | 1.00 | 1.00 |
| 35 | What is the role of the in-game leader (IGL)? | — | 0.77 | — | 1.00 |
| 36 | What is a "split" attack? | — | 0.90 | 0.70 | 1.00 |
| 37 | What is counter-strafing and why does it matter? | — | 0.96 | — | 0.00 |
| 38 | How should CTs retake a site after the bomb is planted? | — | 0.62 | 1.00 | 0.00 |
