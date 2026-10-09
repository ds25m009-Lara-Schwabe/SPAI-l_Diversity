# _l-Diversity_ with Top-Down-Greedy Algorithm

## 1. Project Objective

The goal of this project is to extend an existing k-anonymity implementation wih l-diversity.

The selected algorithm is Top-down Greedy (TDG) from the
kaylode/k-anonymity framework.

The implementation is evaluated on the `cahousing` dataset.

The final comparison will investigate:

- privacy guarantees
- information loss
- classification performance

for the original k-anonymous TDG algorithm and the extended k-anonymous + l-diverse version.

## 2. Dataset

`cahousing`is provided in the existing repository. It contains 20.640 records.

### Quasi-identifiers

The existing framework defined the following attributes as quasi-identifiers:

* longitude
* latitude
* housing_median_age
* median_income
* median_house_value

### Sensitice attribute

`ocean_proximity`is used as the sensitive attribute.

It contains three distinct values:

| Value | Count |
|------:|------:|
| 0 | 4,953 |
| 1 | 9,136 |
| 2 | 6,551 |

The same variable is already defined as the classification target
for `cahousing` in the original framework.

## 3. Privacy Definitions

### k-Anonymity

Each equivalence class must contain at least `k` records:

|EC| >= k

### Distinct l-Diversity

Each equivalence class must contain at least `l` distinct values
of the sensitive attribute:

|distinct(EC[S])| >= l

The implemented extension enforces both constraints.

## 4. Original Top-down Greedy Algorithm

The existing TDG implementation recursively partitions the dataset.

Candidate groups are formed based on distances between records
using Normalized Certainty Penalty (NCP).

The original algorithm prevents partitions smaller than `k` and
uses a balancing procedure when a candidate partition violates
k-anonymity.

## 5. l-Diversity Extension

### 5.1 l-diversity check

A separate helper module `l_diversity.py` was introduced.

It contains functions for:

- counting distinct sensitive values in a partition
- checking whether a partition satisfies distinct l-diversity

### 5.2 Sensitive attribute handling

The TDG implementation internally reorders the dataset so that all
quasi-identifiers appear first.

The sensitive attribute index is therefore dynamically converted
from its original dataset position to its position after column
reordering.

### 5.3 Partition validity

A partition is considered valid when:

1. its size is at least `k`
2. when l-diversity is enabled, it contains at least `l` distinct
   sensitive values

## 6. Reproducibility

The original TDG algorithm contains randomized pair selection.

A fixed random seed was introduced:

`random.seed(42)`

This allows repeated experiments to produce the same partitioning
and therefore the same utility metrics.


## 7. Initial Baseline

Configuration:

- Algorithm: Top-down Greedy
- Dataset: cahousing
- k = 2
- l-diversity disabled
- Random seed = 42

Results:

| Metric | Result |
|---|---:|
| Final TDG partitions | 9,029 |
| NCP | 0.155 |
| CAVG | 1.208 |
| DM | 56,764 |

This serves as the k-anonymity baseline for comparison.

## 8. Initial l-Diversity Attempt

Configuration:

- k = 2
- l = 2
- Sensitive attribute = ocean_proximity

The first implementation rejected a TDG split immediately whenever
one of its two resulting partitions violated l-diversity.

### Result

| Metric | Result |
|---|---:|
| Final partitions | 1 |
| NCP | 1.000 |
| CAVG | 10,320 |
| DM | 426,009,600 |

The complete dataset was retained as a single equivalence class.

Although this result satisfied l-diversity, it caused maximum
generalization and extremely poor data utility.

### Finding

Simply rejecting the first invalid candidate split is too
restrictive.

## 9. Improved l-Diversity Split Strategy

The algorithm was modified to try several candidate TDG splits.

For each partition:

1. Generate a candidate TDG split.
2. Apply the original k-anonymity balancing procedure.
3. Check whether both child partitions satisfy k-anonymity and
   l-diversity.
4. Accept the first valid split.
5. If no valid split can be found after the configured number of
   attempts, retain the parent partition.

Currently:

`MAX_SPLIT_ATTEMPTS = 10`


## 10. First Improved l-Diversity Result

Configuration:

- k = 2
- l = 2
- Sensitive attribute = ocean_proximity
- Random seed = 42

Results:

| Metric | k-anonymity | k + l-diversity |
|---|---:|---:|
| NCP | 0.155 | 0.469 |
| CAVG | 1.208 | 15.426 |
| DM | 56,764 | 4,360,456 |
| Internal TDG partitions | 9,029 | 743 |

The l-diverse version produces substantially more generalization
than standard k-anonymity, which is expected because it introduces
an additional privacy constraint.

## 11. Privacy Validation

The generated anonymized dataset was independently grouped by its
generalized quasi-identifiers.

For k = 2 and l = 2:

| Validation | Result |
|---|---:|
| Equivalence classes | 669 |
| Minimum equivalence-class size | 2 |
| Minimum distinct sensitive values | 2 |
| k-anonymity violations | 0 |
| l-diversity violations | 0 |

The generated dataset therefore satisfies both k-anonymity and
distinct 2-diversity.


## 12. Preliminary Findings

The first experiments indicate that adding l-diversity substantially
increases information loss compared with standard k-anonymity.

The naive approach of simply rejecting an invalid split resulted in
complete generalization.

Allowing TDG to search for alternative valid splits greatly improved
utility while still satisfying both privacy constraints.

Further experiments are required with different k and l values and
with classification performance before drawing final conclusions.


## 13. Next Steps

- Run systematic experiments for different k/l configurations.
- Compare NCP, DM and CAVG.
- Evaluate classification performance.
- Verify k- and l-constraints for every generated dataset.
- Visualize privacy-utility trade-offs.
- Prepare final presentation.