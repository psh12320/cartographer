# Cartographer — Devpost Project Description

**Entropy-guided conversational product search that treats shopping dialogue as active search.**

## How our solution addresses the problem statement

Keyword search works once a shopper already knows the product words. Real shopping conversations are vaguer than that: preferences arrive in pieces, change mid-conversation, and are sometimes deliberately withheld. Cartographer is built around a single observation about how the challenge is actually scored — a session ends the moment the target product appears in the returned list, and the rank at that moment is locked in permanently. Recommending a product you are not confident about is therefore not a free guess; it is an irreversible commitment.

Every design decision follows from that.

### I. Core Architecture — Intent Routing and a Hybrid Pipeline

The agent classifies each message into a **Buying** or **Browsing** track and runs a different retrieval strategy for each.

- **Buying** locks hard constraints. Requirements become destructive filters, but only while metadata coverage stays safe: a constraint narrows the pool only if the intersection stays above a floor, with an escape hatch for the strongest exact matches. This prevents a mis-parsed requirement from filtering the correct product out of existence.
- **Browsing** widens instead of narrowing, weighting category agreement higher and applying intent-fingerprint diversification so an open-ended request returns a spread of options rather than five near-duplicates.

Candidates are unioned from four in-memory routes — SQLite FTS5/BM25 keyword retrieval, a popularity-ordered category prior, exact intent-fingerprint matching, and an optional dense BGE vector route — then fused by route-aware scoring and re-ordered by a learned residual reranker. Retrieval recall is complete — the target is present in the candidate union on turn one in **every** session, so nothing is ever lost by search itself. End-to-end Hit Rate@10 is `0.9990`: exactly one session in a thousand fails to surface it in time, for the reason given below.

### II. Dialog Strategy — Multi-Turn Scenario Evolution

State is explicit and replaceable. Constraints carry strength, source turn, active flag and intent epoch, so the agent can accumulate information incrementally *and* rewrite it.

- **Information accumulation:** each disclosure adds a typed constraint; superseded values are deactivated rather than deleted, preserving an auditable history.
- **Intent Override:** when a shopper corrects themselves, the agent deactivates only the *superseded* preference and keeps everything else they disclosed. An earlier version erased all constraints on override; fixing that to match how the simulated customer actually behaves was worth **+0.0043** on held-out data and lifted Intent Override to MRR 0.985.
- **Proactive Guidance:** when the candidate union is too large to answer with a list, an over-generality cutoff truncates the recommendation to a probe and spends the turn on a structured clarification prompt instead, driving convergence rather than guessing.

Questions are chosen by **expected information gain** over candidate-set entropy, shaped by attribute coverage and the shopper's profile, and the agent never repeats a question the customer has already exhausted.

### III. Self-Evolution — Dynamic Context Programming

- **Runtime adaptation.** Recommendation breadth is re-planned every turn from live evidence rather than a fixed rule. The agent measures the score margin between its first and second candidate; when that margin is thin it declines to widen the list, because converting at rank 2 locks in rank 2 forever while deferring a turn costs only 0.02. We validated the signal before building on it — the leading candidate is correct 40% of the time in the lowest margin quartile versus 72% in the highest. This single mechanism is worth **+0.0118**.
- **Personalized context distillation.** A durable profile store distils each finished session into a long-term record — which attributes a shopper is willing to specify, which categories they explore — and merges it back on their next visit. It stores attribute names, coarse categories and counts only: never product identifiers, never labels, never raw customer text. Across the 1,000 evaluation sessions it builds 64 distinct shopper records, the largest aggregating 157 sessions and learning that this shopper specifies `feature` (199 times) and `material` (159). **It measures at −0.000229 and therefore ships disabled** — see below for why, and for the two bugs that finding uncovered.

## Results

Trained on the 200 public sessions and evaluated with the unmodified official evaluator across all 1,000 labelled sessions (200 public + 800 held-out synthetic):

| Metric | Starter baseline | Cartographer |
|---|---:|---:|
| TechnicalScore | 0.10671 | **0.973062** |
| Hit Rate@10 | 0.125 | **0.9990** |
| MRR | 0.068034 | **0.995875** |
| MTTC | 9.81 | **2.260** |

Held out separately, the 800 synthetic sessions the model never trained on score `0.971891` — the gain there is *larger* than on the in-sample public 200, which is the opposite of overfitting. Per scenario: Boundary MRR `1.0000`, Buying `0.9988`, Browsing `0.9950`, Intent Override `0.9892`. Every figure is reproducible with `python3 -m cartographer.reproduce`.

One honest caveat rather than a rounded-up headline: **Hit Rate is `0.9990`, not perfect.** The confidence gate deliberately trades a little recall for rank — by declining to widen a list it is unsure of, it slows the growth of the "already shown" set that normally pushes past bad candidates, and one session in a thousand never converts. The relationship is monotonic in the threshold, so this is a real mechanism and not noise. Releasing full depth two turns earlier recovered one of the two sessions this originally cost *and* raised the score; the last one resisted every variant we tried and sits at the resolution limit of a 1,000-session benchmark.

Inference is CPU-only, offline, deterministic, and reports **zero tokens at $0 cost**. Turn p95 latency is around 250 ms against a 750 ms budget.

## Development tools used

VS Code, Git/GitHub, Python 3.12 on Linux, a Gradio dashboard built for diagnosis, and an experiment harness that scores every candidate change through the unmodified evaluator.

## APIs used

**None.** No external API is called at inference or training time. There is no LLM in the loop, no network dependency, and no paid service. This was a deliberate choice: the challenge rewards conversational precision, and a deterministic, inspectable pipeline proved both cheaper and easier to debug than an opaque one.

## Libraries and frameworks used

The **deterministic runtime uses only the Python standard library** — `sqlite3` (FTS5), `json`, `re`, `math`, `hashlib`. This is the entire submitted agent.

Optional and development-only: `numpy` and `sentence-transformers`/`PyTorch` for the offline BGE embedding route (evaluated and left disabled, see below), and `gradio` for the diagnostic dashboard.

## Datasets and assets used

- The organizer's frozen 50,000-product catalog derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), verified by SHA-256 and never redistributed.
- The 200 labelled public development sessions.
- `synthetic_800_v1.jsonl` — an 800-session held-out set we generated to the official scenario mix, sharing **no target product and no sample identifier** with the public sessions, used purely as an out-of-sample check.
- `BAAI/bge-small-en-v1.5` embeddings, built offline and checksum-verified.

## Long-term personalization: a measured negative, and two bugs it exposed

We had assumed this capability was unmeasurable, on the reasoning that every evaluation session is a fresh user. That was wrong, and checking rather than assuming is what surfaced it: the 1,000 sessions contain only **64 distinct profile signatures**, the largest covering 130 sessions. The competition profiles are shopper *archetypes*, not unique identities, so a returning-shopper record actually fires on 94% of sessions and the feature is directly testable.

| Configuration | TechnicalScore | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Profile memory off (shipped) | **0.972982** | 0.9990 | 0.995875 | 2.2640 |
| Profile memory on | 0.972753 | 0.9990 | 0.995708 | 2.2730 |

The first measurement returned a delta of **exactly 0.000000**, which is the tell that mattered: a feature that is genuinely running essentially never scores identically to its absence. Investigating that instead of accepting a convenient "no harm" result exposed two real defects.

1. **Distillation never ran at all.** The official evaluator allocates a fresh `uuid` per conversation, so keying the fold-in on a repeated `session_id` silently matched nothing. Our unit test passed only because it reused one identifier. The fix distils every finished conversation exactly once regardless of naming, and a regression test now mirrors the evaluator's naming exactly.
2. **Shopper identity drifted.** Recall enriches the incoming profile with remembered attributes, so the *next* distillation computed a different key from the enriched profile and fragmented one shopper into many records. Identity is now captured from the caller's original profile, before any enrichment.

With both fixed the feature works — and still costs `0.000229`. The reason is instructive rather than disappointing: the agent already converges at MRR `0.9957` in about 2.3 turns, so a prior over which attributes a shopper tends to specify has almost nothing left to improve, and the extra tags occasionally nudge question selection away from the better choice. It ships **off by default**, as a working and tested capability whose value would appear in a deployment with genuinely returning shoppers rather than 64 archetypes.

## What we learned

The most valuable engineering result was negative. We rejected dense retrieval on three independent instruments: an end-to-end grid spanning less than three sessions of resolution, a fixed-message replay showing every dense variant *degraded* turn-one ranking, and a dense-retrained cross-validation gaining an order of magnitude less than the promotion gate. The reason is that this customer discloses requirements as verbatim substrings of the product's own text, which exact matching resolves precisely and cosine similarity blurs.

We also learned the reranker was not the bottleneck: giving it **8× more training data changed the score by −0.0002**, and richer objectives and extra features both made it worse. Every meaningful gain came instead from correcting agent *logic* — how state is rewritten on an override, how many requirements a question can harvest, and when the agent should decline to answer.

## Limitations and what we would improve

- Attribute extraction is tuned to an English clothing catalog and would need new taxonomies elsewhere.
- Long-term personalisation is real, working and tested, but measures at `-0.000229` on this benchmark because the agent already converges in about 2.3 turns. It ships disabled and is reported as a capability, not a score claim.
- Intent Override sessions are bounded by the simulator: they cannot convert before the override arrives, which caps the achievable score at ~0.9926 rather than 1.0.
- With more time: a compact local classifier for subtle paraphrases, per-scenario question policies, and online measurement of conversion lift.
