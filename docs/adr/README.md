# Architecture decision records

Each record describes one design decision: the situation that forced it, what we chose, the alternatives we rejected, and the evidence. Numbers come from our held-out folds unless marked as leaderboard (LB) results.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-blocking-then-learned-ranking.md) | Multi-channel blocking plus a learned ranker | Accepted |
| [0002](0002-hard-country-partition.md) | Hard country partition, country as an open set | Accepted |
| [0003](0003-gradient-boosting-as-core-matcher.md) | Gradient-boosted trees as the core matcher | Accepted |
| [0004](0004-list-context-stacking.md) | Stacked second stage over the whole candidate list | Accepted |
| [0005](0005-exact-expected-f05-decision.md) | Exact expected-F0.5 decision | Accepted |
| [0006](0006-test-like-validation.md) | Grouped cross-fitting, sealed fold and decoy injection | Accepted |
| [0007](0007-cross-encoders-on-uncertain-band.md) | Small cross-encoders on the uncertain band only | Accepted |
| [0008](0008-two-pipelines-and-second-opinion.md) | Two independent pipelines and a second-opinion correction | Accepted |
| [0009](0009-label-free-validation-for-france.md) | France: label-free validated rules | Accepted |
| [0010](0010-rules-as-explicit-change-lists.md) | Rules shipped as explicit change lists | Accepted |

Format: Context → Decision → Alternatives considered → Consequences.
