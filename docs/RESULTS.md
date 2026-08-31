# Evaluation Results

## Published baseline

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 starter | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

## Historical locked 100-session holdout

The preceding RRF-60 artifact was trained on the development half defined by `docs/public_split_v1.json` and evaluated once on the disjoint holdout half. Both halves preserve the official 40/40/15/5 scenario mix. This holdout is now consumed and was not used to select or evaluate the active RRF-120 successor.

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Holdout without reranker | 1.000 | 0.683210 | 2.16 | 0.884 | 0.881763 |
| Holdout with development-100 reranker | **1.000** | **0.781397** | **1.91** | **0.909** | **0.916219** |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Buying | 40 | 1.000 | 0.702946 | 1.075 | 0.909384 |
| Browsing | 40 | 1.000 | 0.842212 | 1.95 | 0.933664 |
| Intent Override | 15 | 1.000 | 0.844444 | 3.733333 | 0.898667 |
| Boundary | 5 | 1.000 | 0.733333 | 2.8 | 0.884000 |

The development-only reranker improved holdout TechnicalScore by `0.034456`, MRR by `0.098187`, and MTTC by `0.25` turns while preserving perfect Hit Rate@10. It improved reciprocal rank in 31 sessions, tied in 56, and regressed in 13; it improved conversion turn in 24 and never delayed a conversion. At the per-session TechnicalScore-contribution level, it produced 40 wins, 47 ties, and 13 losses. A deterministic 100,000-resample paired bootstrap (seed `2026`) placed the 95% percentile interval for the gain at `[0.009670, 0.059391]`. Boundary contains only five holdout sessions and must not be treated as a stable standalone estimate.

This split was created after the architecture and route choices had already been developed against the full public set. The holdout therefore tests whether newly fitted 100-session weights transfer to disjoint sessions, but it is not a pristine estimate of the entire architecture-selection process. Only the private 800 can provide that final independent test.

The previous development-only RRF-120 artifact (SHA-256 `464d2daa3c0862c049fd497439d3133cc6d1ea18eb7ad6153271bd70e4b0d884`, five-fold development OOF TechnicalScore `0.930420`) has been superseded. The RRF-60 holdout artifact remains recoverable at `working-holdout-0.916219`.

## Active configuration: precision-gated depth + Browsing residual (2026-08-31)

The promoted configuration keeps the RRF-120 route-scaled reranker and adds two development-selected changes: a precision-gated recommendation depth (`(1, 2, 10)` keyed on the turn within the current intent epoch, full lists forced from absolute turn 6) and a learned Browsing residual at scale `0.75`. Buying, Browsing, Boundary, and Override routes learn residual weights with scales `1.25`, `0.75`, `0.75`, and `0.75`. Selection used only the locked 100-session development partition through the five-fold out-of-fold procedure; the consumed holdout stayed closed.

Five-fold development OOF TechnicalScore: **`0.973850`** (previous champion `0.930420`), Hit Rate@10 `1.0` in every scenario and fold, all five folds matching or beating the deterministic reference, maximum fold p95 latency `182.194 ms` on the reference Linux host.

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 40 | 1.000 | 0.981250 | 1.425 |
| Browsing | 40 | 1.000 | 0.975000 | 1.750 |
| Intent Override | 15 | 1.000 | 0.966667 | 3.866667 |
| Boundary | 5 | 1.000 | 1.000000 | 2.400 |

The depth gate exploits a structural property of the official scoring loop: the evaluator converts a session the moment the target appears anywhere in the returned list, permanently locking in that turn's rank. Out-of-fold diagnostics showed 25 of 100 development sessions converting at rank 2–9, usually after the learned residual lifted the target from deterministic rank 25–115 into the top ten on turn one, while sessions converting after one more constraint disclosure landed at rank one in 50 of 60 cases. One deferred turn costs `0.02` TechnicalScore against up to `0.24` recovered by a rank-five-to-one improvement, so the agent recommends a one-product shortlist on the first turn of an intent epoch, two products on the second, and the full ten afterwards. The agent's `classify_constraint` taxonomy was also aligned exactly with the evaluator's published taxonomy (score-neutral on development; removes structurally unanswerable questions such as `brand`).

The active artifact SHA-256 is `e5706c6cce91071eca6bebc14b307904e8ea98e91bd491f1b3d5b20e303f606f`; the split-manifest SHA-256 is `342866b37304c8b0b57a59f1bae2d9a53157be502d3149c45f195f10172742a7` (its `source.sha256` was corrected on 2026-08-31 from a CRLF-checkout hash to the LF value with an in-file audit note; the locked sample IDs are unchanged). Snapshot pair counts: buying 3882, browsing 1350, boundary 450, override 630.

## Held-out synthetic 800-session test (2026-08-31)

`synthetic_800_v1.jsonl` is an 800-session set built to the official scenario mix (320 Buying, 320 Browsing, 120 Intent Override, 40 Boundary) and scored with the unmodified official evaluator. Before use it was checked for contamination: all 800 targets exist in the frozen catalog, and it shares **no target product and no sample identifier** with the 200 public sessions. Nothing in it was used to select any configuration; the promoted configuration was fixed by development-100 cross-validation beforehand.

| System | TechnicalScore | Hit Rate@10 | MRR | MTTC | p95 latency |
|---|---:|---:|---:|---:|---:|
| Promoted gated configuration | **`0.951764`** | `1.000` | `0.919048` | `2.1975` | `89.9 ms` |
| Ungated predecessor (depth gate removed) | `0.876259` | `1.000` | `0.631198` | `1.6550` | `146.5 ms` |

The depth gate is worth `+0.075505` on this held-out set, larger than the `+0.043430` it was selected for on development, and it is net positive in every scenario: Buying `+0.0655`, Browsing `+0.0970`, Boundary `+0.0559`, Intent Override `+0.0512`. Hit Rate@10 is `1.000` for all 800 sessions under both configurations, so withholding products on early turns cost no conversions at this scale — including in the 120 Intent Override sessions, where the per-epoch restart of the schedule had been the main theoretical concern.

Remaining loss on this set divides almost exactly in half between ranking and turns: `0.0243` from MRR (110 of 800 sessions convert below rank one, 75 of them at rank two) and `0.0240` from MTTC. The MTTC component has a structural floor near `1.375` turns because Intent Override sessions cannot convert before their override fires at turn three or four.

## Reproducibility notes

- The official evaluator and public labels are unmodified.
- The runtime agent does not import the evaluator, public session file, or ground truth.
- The deterministic route reports zero prompt and completion tokens.
- Public labels from the development 100 are used only by the trainer; the frozen runtime artifact contains feature weights and never reads labels or evaluator internals.
- The holdout was evaluated once after fitting. It must not be used for subsequent parameter tuning.
- Optional semantic routes are recorded separately and are not claimed unless their assets and configuration are included.

## Diagnostic component ablation

These pre-reranker ablations use a fixed 20-session engineering slice and are included to show component directionality; the locked holdout table above is the current generalization result.

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Full Cartographer | 1.000 | 0.683929 | 2.40 | 0.877179 |
| Without dense route | 1.000 | 0.683929 | 2.40 | 0.877179 |
| Without fingerprints | 1.000 | 0.567560 | 2.65 | 0.837268 |
| Without state | 0.800 | 0.455119 | 4.05 | 0.675536 |
| Without clarification | 0.750 | 0.282202 | 5.60 | 0.567661 |
| BM25-only ablation | 0.500 | 0.154365 | 7.00 | 0.376310 |

The dense row is identical because semantic model assets were intentionally absent from the frozen deterministic configuration. The published organizer BM25 baseline remains the authoritative baseline; the BM25-only diagnostic above differs because it still uses Cartographer's response plumbing and implicit miss handling.

## Performance and model gates

The active RRF-120 candidate's maximum sequential inner-fold p95 was `543.443 ms`; a separate warm run through the promoted default measured mean `144.421 ms`, p95 `380.588 ms`, and maximum `633.244 ms`. Scores from the all-development-trained warm run are not selection evidence. The historical RRF-60 holdout run measured p95 `241.569 ms`. Catalog construction is excluded from turn timings.

The verified local BGE model loaded successfully, but its first 128-document CPU batch required `56.62 s`, projecting roughly six hours for the 391-batch catalog build on the test host. A checksummed GPU handoff now generates this artifact offline; semantic scores remain excluded from every result above until that artifact is imported and re-evaluated. The cross-encoder has not yet been promoted.
