# Demo Video Script

Target length: 2–3 minutes.

## 1. The problem — 20 seconds

Show the weak BM25 baseline and explain that shoppers rarely provide every useful keyword at once. Highlight the three evaluated outcomes: finding the target, ranking it highly, and doing so in fewer turns.

## 2. Architecture — 30 seconds

Show the README architecture diagram. Explain the two routes, participant-visible intent fingerprints, dynamic context state, hybrid retrieval, and information-gain question policy. Emphasize local CPU execution, zero API cost, and no catalog modification.

## 3. Live multi-turn trace — 60 seconds

Open the dashboard's **Session replay** and select an Intent Override session. Compare the full agent with only the precision gate removed. Call out the route, candidate count, entropy, selected clarification attribute, information gain, returned depth, gate reason, intent-epoch change after “Actually, ignore…”, retention of unrelated disclosed constraints, and final target rank. Then rerun with only the learned reranker removed and use the dominant-score-contribution column to explain the rank movement.

## 4. Results and ablations — 30 seconds

Show the dashboard's **Component value lab**. Compare the full agent with the published starter and show reranker, gate, fingerprints, state, and clarification ablations. Include overall and scenario TechnicalScore, p95 latency, memory from the frozen benchmark, zero token usage, zero API cost, and offline fallback. Finish on the **Report & video evidence** checklist.

## 5. Impact — 20 seconds

Close with the generalization story: the same active-search loop can power jobs, travel, real estate, procurement, and knowledge discovery. A good copilot does not merely answer—it decides what is most valuable to learn next.
