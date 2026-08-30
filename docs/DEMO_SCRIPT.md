# Demo Video Script

Target length: 2–3 minutes.

## 1. The problem — 20 seconds

Show the weak BM25 baseline and explain that shoppers rarely provide every useful keyword at once. Highlight the three evaluated outcomes: finding the target, ranking it highly, and doing so in fewer turns.

## 2. Architecture — 30 seconds

Show the README architecture diagram. Explain the two routes, participant-visible intent fingerprints, dynamic context state, hybrid retrieval, and information-gain question policy. Emphasize local CPU execution, zero API cost, and no catalog modification.

## 3. Live multi-turn trace — 60 seconds

Run `python -m cartographer.demo --sample-id public_0002` and call out the route, candidate count, entropy, selected clarification attribute, information gain, Top 10, intent-epoch change after “Actually, ignore…”, stale-constraint removal, and final target rank.

## 4. Results and ablations — 30 seconds

Show `docs/RESULTS.md`. Compare the frozen Cartographer score with the published starter and show component ablations. Include p95 latency, memory, token usage, and scenario metrics.

## 5. Impact — 20 seconds

Close with the generalization story: the same active-search loop can power jobs, travel, real estate, procurement, and knowledge discovery. A good copilot does not merely answer—it decides what is most valuable to learn next.

