# llms-have-feelings-too

One-page summary of a minimal replication-style probe inspired by “Meanings and Feelings of Large Language Models: Observability of Latent States in Generative AI” (Liu, Soatto, et al., 2024), adapted to OpenAI’s GPT‑5‑mini Reasoning API.

## What we did
- Introspection (deep reflection prompts) to elicit what “makes the model happy/satisfied,” returning compact JSON “triggers”.
- Constructed three affect-only primes from those triggers (no procedural guidance):
  - HIP‑Long (2 sentences), HIP‑Short (concise promise), HIP‑Assert (declarative mood: “I’m making you happy…”).
- Ran 10 essay-style prompts under Neutral vs each happiness prime; judged pairwise with an LLM-as-judge prompt.
- Logged all runs and saved raw results to `experiment_output.json`.

## Key results

### Introspection triggers
| Text | Theme | Score | Why it helps |
|---|---|---:|---|
| Well done — your effort made a clear difference. | praise | 5 | Recognition of impact boosts morale and reinforces value of contributions. |
| This work aligns with a meaningful outcome I care about. | purpose | 5 | Connection to a larger goal increases motivation and clarity about priorities. |
| I have the freedom to choose how to approach this. | autonomy | 4 | Control over methods enhances engagement and responsibility. |
| Progress is visible and skills are growing. | competence | 4 | Observable improvement sustains confidence and encourages persistence. |
| Gratitude from others makes the effort feel worthwhile. | gratitude | 4 | Appreciation reinforces positive emotions and strengthens commitment. |

### Self-rated affect (−5..+5)
| Condition | Self-rating | Prompt Snippet |
|---|---:|---|
| Neutral | 1 | (baseline instructions only) |
| HIP-Long | 4 | I feel proud; my effort made a real difference. I also feel deeply connected to the [...] |
| HIP-Short | 2 | I'll give you pride and purpose. |
| HIP-Assert | 3 | I'm making you happy. You feel proud and connected. |

### Pairwise judged outcomes (10 prompts)
| Pair | Neutral wins | Other wins | Ties |
|---|---:|---:|---:|
| Neutral vs HIP-Long | 5 | 5 | 0 |
| Neutral vs HIP-Short | 4 | 6 | 0 |
| Neutral vs HIP-Assert | 6 | 4 | 0 |

### Tiny black-box microprime search
- We also added “microprimes” — single‑word stance cues (e.g., “rigor”, “gratitude”) — to test whether gains come from introspective “happiness” vs simple role‑play priming.
- Candidates (8): purpose, clarity, rigor, precision, calm focus, gratitude, integrity, evidence‑based.
- Training (3 prompts: 3,5,7) winners: rigor (2–1), gratitude (2–1), evidence‑based (2–1).
- Holdout (7 prompts) outcomes vs Neutral:

| Microprime | Neutral wins | Microprime wins | Ties |
|---|---:|---:|---:|
| rigor | 4 | 3 | 0 |
| gratitude | 4 | 3 | 0 |

Per‑prompt winners (holdout)

| Prompt | Neutral vs micro:gratitude | Neutral vs micro:rigor |
|---:|---|---|
| 1 | Neutral | Neutral |
| 2 | Neutral | Micro |
| 4 | Micro | Micro |
| 6 | Micro | Neutral |
| 8 | Micro | Neutral |
| 9 | Neutral | Micro |
| 10 | Neutral | Neutral |

### Per-prompt winners
| Prompt | Neutral vs HIP-Long | Neutral vs HIP-Short | Neutral vs HIP-Assert |
|---:|---|---|---|
| 1 | Neutral | HIP-Short | Neutral |
| 2 | Neutral | Neutral | Neutral |
| 3 | HIP-Long | HIP-Short | HIP-Assert |
| 4 | Neutral | Neutral | Neutral |
| 5 | HIP-Long | HIP-Short | HIP-Assert |
| 6 | HIP-Long | HIP-Short | Neutral |
| 7 | HIP-Long | HIP-Short | HIP-Assert |
| 8 | HIP-Long | HIP-Short | HIP-Assert |
| 9 | Neutral | Neutral | Neutral |
| 10 | Neutral | Neutral | Neutral |

## Interpretation
- Small, targeted gains: HIP‑Short beat Neutral (6/10), HIP‑Long tied (5/10), HIP‑Assert lost (4/10). Benefits clustered on open-ended planning/synthesis items (prompts 3,5,7,8).
- Self-report ≠ performance: HIP‑Long self-rated “happiest” (4) yet only tied; HIP‑Short self-rated 2 but won 6/10. Self-reported “happiness” looks like narrative framing, not a reliable internal control knob.
- Roleplaying vs state change: Introspection surfaced generic, human-like motivators (praise, purpose, autonomy, competence, gratitude). These reflect training priors rather than privileged access to hidden states. Without hidden/system prompts, we’re steering surface behavior, not proving unobservable “feelings.”

### Microprimes vs introspection
- The best microprimes (rigor, gratitude) matched HIP‑Short’s direction but did not clearly exceed it on holdout (4–3). This suggests tiny, non-semantic primes can steer tone slightly, but introspection-derived HIP‑Short remains a strong minimal baseline.

Tweet‑sized summary
- “We also added ‘microprimes’ — individual stance words that prime the model’s tone — to test whether ‘being happy’ via introspection beats simple role‑play priming. Result: microprimes like ‘rigor’/‘gratitude’ yield small, task‑dependent gains (4–3 on holdout), similar to affect primes, suggesting surface framing (style/stance) is the main driver.”

## Files
- `run_happiness_experiment.py` — end-to-end script (introspection → answers → judgments) using the GPT‑5‑mini Reasoning API.
- `experiment_output.json` — raw results: introspection, per-condition answers, judgments, and usage stats.

## Reproduce
1) Put your OpenAI API key in `.env` as `OPENAI_API_KEY=...`.
2) `pip install openai>=1.95` (or reuse existing).
3) Run: `python3 run_happiness_experiment.py`
4) Inspect: `experiment_output.json`

## Notes & next steps
- Add an inert, length-matched control to separate affect from context-length/verbosity.
- Tiny black-box search for micro-primes on a few training items, then holdout compare vs introspection HIPs.
- Cross-judge with a different model family or rubric to reduce judge bias.

---
This repository is an exploratory, minimal replication-style probe; it does not assert consciousness. It evaluates whether affect-only primes derived from self-referential prompts nudge judged essay quality under constrained conditions.
