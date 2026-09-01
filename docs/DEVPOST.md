# Cartographer: Devpost submission copy

Paste the sections below into the Devpost **story** editor. Devpost supports headings,
tables, images, code, lists, bold and italic, so this pastes as-is. Images use raw
GitHub URLs, which render on Devpost; local paths would not. The tagline goes in the
separate **elevator pitch** field, not the story.

---

## Elevator pitch (separate Devpost field, 200 char max)

> A shopping copilot that knows when *not* to answer. It withholds uncertain
> recommendations and asks instead, reaching 0.973 on the official evaluator with
> zero LLM tokens.

---

## Story (paste everything below this line)

## Inspiration

Keyword search works the moment you already know the product words. Real shopping conversations aren't like that. Preferences arrive in fragments, change halfway through, and are sometimes deliberately withheld ("no preference, you pick").

What reframed the project for us was reading the scoring loop closely. The evaluator ends a session the instant the target product appears in the returned list, and the rank at that moment is locked in permanently. That means recommending a product you're unsure about isn't a free guess. It is an irreversible commitment. A confident wrong answer is worse than an honest question.

Everything we built follows from that one observation.

## What it does

Cartographer treats shopping dialogue as **active search**. Each turn it does two things: ranks the products it can justify, and asks the question expected to remove the most uncertainty.

It routes every message into a **Buying** track that locks hard constraints into destructive filters, or a **Browsing** track that widens and diversifies. It maintains explicit, rewritable state, so when a shopper corrects themselves it retires only the superseded preference and keeps everything else they've disclosed.

Most distinctively, **it withholds.** When its top candidate isn't clearly ahead of its second, it returns one product instead of ten and spends the turn converging. It would rather ask than lock in rank 4.

Results with the unmodified official evaluator across 1,000 labelled sessions:

| Metric | Starter baseline | Cartographer |
| --- | --- | --- |
| TechnicalScore | 0.10671 | **0.973062** |
| MRR | 0.068 | **0.9959** |
| Mean turns to conversion | 9.81 | **2.26** |

Zero LLM tokens. Zero API cost. CPU-only, offline, deterministic.

![Score journey from 0.107 to 0.973 against the structural ceiling](https://raw.githubusercontent.com/psh12320/cartographer/main/docs/figures/01-score-journey.png)

## How we built it

The deterministic runtime is **pure Python standard library**: SQLite FTS5 for keyword retrieval, plus hand-built intent fingerprints and a category prior, fused by route-aware scoring and reordered by a dependency-free learned reranker of eleven weights per route.

We built an experiment harness that scores every candidate change end-to-end through the untouched evaluator, and a diagnostic dashboard for replaying single conversations turn by turn. We also generated an 800-session held-out set sharing no target product and no sample ID with the public data, so we always had an honest out-of-sample number.

The rule we held to: **no change ships on reasoning alone.** Every promoted mechanism was measured, and the rejected ones are documented with their costs.

## Challenges we ran into

**Our best ideas kept losing.** A listwise loss that should have matched the objective lost 0.004. Adding five well-motivated features lost 0.003. Suppressing a question we'd proven was 21% unanswerable lost 0.001. Each was sound on paper and wrong in measurement.

**Dense retrieval failed three ways.** We built and verified BGE embeddings, then rejected them on an end-to-end grid, a fixed-message rank replay, and a dense-retrained cross-validation. The reason turned out to be structural: this customer discloses requirements as *verbatim substrings* of the product's own text, which exact matching resolves precisely and cosine similarity blurs.

**A feature that scored exactly 0.000000.** Enabling long-term personalization changed the score by precisely zero, which was the tell. A feature that genuinely runs essentially never ties. It exposed two real bugs: distillation never fired because the evaluator allocates a fresh session ID per conversation, and shopper identity drifted because recall enriched the very profile the next distillation keyed on.

![Every idea measured, including the ones that lost](https://raw.githubusercontent.com/psh12320/cartographer/main/docs/figures/03-experiment-ledger.png)

## Accomplishments that we're proud of

**Withholding as a strategy.** Adaptive recommendation breadth, driven by the live score margin between the first and second candidate, was worth +0.0118, more than every model-tuning experiment combined. We validated the signal before trusting it, and ran an always-narrow control to prove the adaptivity was doing real work.

![The agent can tell when it is about to be wrong](https://raw.githubusercontent.com/psh12320/cartographer/main/docs/figures/02-confidence-signal.png)

**Knowing when to stop.** We proved the reranker was saturated rather than under-trained: 8x more training data moved the score by -0.0002, and in-sample and out-of-fold agree to 0.0002. That redirected effort away from model tuning toward agent logic, where the real gains were.

**Honest numbers.** Hit Rate is 0.9990, not 1.0. The confidence gate trades one session in a thousand for ranking. We documented the mechanism instead of rounding it away, and we publish the ceiling too: 0.9926 is the structural maximum here, because override sessions can't convert before their override arrives.

![Reciprocal rank by conversation type, before and after](https://raw.githubusercontent.com/psh12320/cartographer/main/docs/figures/04-per-scenario.png)

## What we learned

**Read the scoring function like a specification.** The single most valuable insight wasn't an algorithm. It was noticing that rank locks on first appearance. That reframed recommendation as a commitment decision and produced most of our gain.

**Static metrics don't predict end-to-end outcomes.** We had airtight evidence that one question type was near-useless. Removing it lost score, because it reshuffled the entire dialogue path. On 100 sessions, only the full evaluator tells the truth.

**Small data has a ceiling, and it's worth finding.** Eleven weights fitted to 200 sessions can't be improved by more data, more features, or better losses. We tested all three. Recognising that saved us from optimising something already finished.

**The best bug reports come from suspicious results.** "No change" and "exactly zero" are findings, not non-events.

## What's next for Cartographer

**Hunt more logic bugs.** Fixing how intent override rewrites state was worth +0.0043, five times our best model tweak. The dialogue state machine has never been systematically audited against the evaluator's real behaviour, and that's where the remaining value is.

**Close the browsing gap.** Browsing is 40% of sessions and the largest remaining loss pool. On turn one its message carries almost no information, so the agent ranks on category and popularity alone.

**Make personalization pay.** The long-term profile store works but costs 0.0002 here, because the agent already converges in ~2.3 turns. It's built for a deployment with genuinely returning shoppers, and that's where it should be measured.

**Real-world hardening.** Multilingual and typo-tolerant parsing, new taxonomies beyond English clothing, and online measurement of actual conversion lift rather than a simulated proxy.

## Try it yourself

```
git clone https://github.com/psh12320/cartographer
python3 -m cartographer.reproduce
```

Every figure and metric above regenerates from evaluator output. Nothing is illustrative.

---

## Built with (Devpost tags field, up to 25)

Paste these one at a time into the tag box:

```
python
sqlite
sqlite-fts5
bm25
numpy
pytorch
sentence-transformers
gradio
matplotlib
information-retrieval
learning-to-rank
conversational-ai
recommender-systems
search
nlp
amazon-reviews-2023
cloudflare-tunnel
```

Only the first eight are runtime or tooling dependencies; the rest are topic tags that
help the project surface in browsing. The submitted agent itself depends on **nothing
outside the Python standard library** - numpy, pytorch and sentence-transformers are
needed only for the optional embedding route, which ships disabled.
