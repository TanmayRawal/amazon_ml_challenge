# ADR 0002: Hard country partition, country as an open set

**Status:** Accepted

## Context

Training contains the US and India; the test set adds France, which has no labels. In the ground truth, 0 of 7,638,365 true pairs cross a country boundary, and names that match only in another country occur for 0.01–0.04% of records.

## Decision

- Every stage (blocking, statistics such as word frequency, ranking, decision) runs **within a country**.
- The pipeline loops over whatever country values exist in the data. Nothing is hard-coded, so France is processed exactly like the US and India.
- **Country is never a model feature.** A model that learned "US" or "India" as an input would have no sensible value for France.
- Word-frequency statistics are computed on each country's own unlabelled pool, so "common" means common in that country.

## Alternatives considered

- **Global blocking across countries.** Larger blocks and more false candidates, for no recall gain.
- **Country as a feature.** Would let the model memorise country-specific base rates that do not transfer to France.

## Consequences

- No cross-country pair can ever be predicted, which matches the ground truth exactly.
- Memory stays bounded, because each country is processed separately.
- French-specific behaviour has to come from generic features or from explicitly validated rules ([ADR 0009](0009-label-free-validation-for-france.md)), not from a France-trained model.
