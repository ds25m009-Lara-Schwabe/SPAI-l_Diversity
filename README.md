<a href="https://doi.org/10.5281/zenodo.22748399"><img src="https://zenodo.org/badge/370051182.svg" alt="DOI"></a>


# Data anonymization using k-Anonymity 
## ✔️ Experiments
- Provides 4 k-anonymization methods: 
  - Datafly
  - Topdown Greedy
  - Topdown Greedy with l-diversity
  - Cluster-based (k-member)
- Implements 3 anonymization metrics: 
  - Equivalent Class size metric (CAVG)
  - Discernibility Metric (DM)
  - Normalized Certainty Penalty (NCP)
- Implements 3 classification models: 
  - Random Forests 
  - Support Vector Machines 
  - K-Nearest Neighbors


## l-Diversity for Top-Down Greedy (SPAI 2026, Exercise 1.2)
We added l-diversity to Top-Down Greedy (`topdown`). The new method `topdown_ldiv` only keeps a split if both halves have at least `k` records and are l-diverse in the sensitive attribute (SA). If one half is not valid, records from the other half that bring in missing SA values are moved over (nearest first, as long as the other half stays valid). If that does not work the partition is not split. Supported types ([Machanavajjhala et al. 2007](https://dl.acm.org/doi/10.1145/1217299.1217302)):

| `--ldiv` | condition for every EC |
|---|---|
| `distinct` (default) | at least `l` different SA values |
| `entropy` | entropy of the SA values >= log(`l`) |
| `recursive` | r<sub>1</sub> < c (r<sub>l</sub> + ... + r<sub>m</sub>), set c with `--c` |

```
python anonymize.py --method=topdown --k=10 --dataset=adult
python anonymize.py --method=topdown_ldiv --k=10 --l=3 --dataset=adult
python anonymize.py --method=topdown_ldiv --k=10 --l=4 --ldiv=entropy --dataset=adult
python anonymize.py --method=topdown_ldiv --k=10 --l=3 --ldiv=recursive --c=2 --dataset=adult
```
- Default SA per dataset is in `datasets/__init__.py` (`adult`: occupation, `italia`: disease, others: class attribute). The SA is removed from the QIs.
- The whole table has to be l-diverse, otherwise the run stops with an error.
- Every run prints the number of ECs, min. distinct / entropy l, homogeneous ECs and ECs violating `--l`.
- Output: `results/{dataset}/topdown_ldiv/{dataset}_anonymized_{k}_l{l}.csv`

Experiments, demo and tests:
```
python compare_topdown.py --dataset=adult              # experiments/adult_topdown_ldiv_results.csv + experiments/figures/
python compare_topdown.py --dataset=adult --plot_only
python demo_topdown.py --dataset=italia --k=2 --l=3   # ECs of both methods, experiments/italia_demo_k2_l3.md
python -m unittest discover tests
```

Changed files: `algorithms/top_down_greedy/` (l-diversity check, split repair, fixed seed, reset of `IS_CAT`), `metrics/ldiv.py`, `anonymize.py` (`--l`, `--sa`, `--ldiv`, `--c`), `datasets/__init__.py`, `models/` (fixes in the classifier evaluation: leaf values were dropped from the features, F1 ignored classes that were never predicted, random label order, no scaling for SVM).

Results on `adult` (k = 10, SA = occupation):

| | NCP | ECs | min. distinct l | ECs < 3 values | F1 KNN / SVM / RF |
|---|---|---|---|---|---|
| original data | | | | | 0.762 / 0.823 / 0.814 |
| `topdown` | 0.091 | 1,784 | 1 | 45 | 0.762 / 0.818 / 0.811 |
| `topdown_ldiv` distinct l = 3 | 0.092 | 1,758 | 3 | 0 | 0.763 / 0.816 / 0.811 |
| `topdown_ldiv` distinct l = 6 | 0.112 | 1,409 | 6 | 0 | 0.762 / 0.817 / 0.812 |
| `topdown_ldiv` entropy l = 4 | 0.125 | 1,439 | 4 | 0 | 0.763 / 0.816 / 0.810 |

Report and slides are in `report/`.

## 📖 Reports
- Report and slides of the original toolbox (Vietnamese, by the toolbox authors): [report](https://github.com/kaylode/k-anonymity/blob/main/demo/report.pdf), [slides](https://github.com/kaylode/k-anonymity/blob/main/demo/slides.pdf)
- Report and slides of the Top-Down Greedy l-diversity extension (Exercise 1.2): [report/](./report/)

## Folder Structure
- A dataset must comes with a .csv file contains features information and a hierarchy folder which contains predefined generalization hierarchies for its QID attributes. 
```
this repo
│   anonymize.py
|
└───data  
│   │
│   └───adult
│       │   adult.csv
│       └───hierarchies
│       │     adult_hierarchy_workclass.csv
│       │     ....
```

- Here is an example for a generalization hierarchy of the 'workclass' attribute from ADULT dataset, described in ```adult_hierarchy_workclass.csv```, which is a csv file using **";" as delimiter**
```
Private;Non-Government;*
Self-emp-not-inc;Non-Government;*
Self-emp-inc;Non-Government;*
Federal-gov;Government;*
Local-gov;Government;*
State-gov;Government;*
Without-pay;Unemployed;*
Never-worked;Unemployed;*
```

which describes this tree:

<div align="center"><img width="450" alt="screen" src="demo/adult_workclass.PNG"></div>

-------------------------------------------------------------

## 🌟 Executing
To anonymize dataset, run:
```
python anonymize.py --method=<model_type> --k=<k-anonymity> --dataset=<dataset_name>
```
- **model_type**: [topdown | topdown_ldiv | cluster | datafly]
- **l**, **sa**, **ldiv**, **c**: l-diversity options for `topdown_ldiv` (see above)
- **dataset_name**: [adult | cahousing | cmc | mgm | informs | italia]

Results will be in ```results/{dataset}/{method}``` folder

To run evaluation metrics on every combination of algorithms, datasets and value k, run:
```
python visualize.py
```

Results will be in ```demo/{metrics.png, metrics_ml.png}``` 

## K-Anonymity examples

| Before anonymization | After anonymization with k = 2 |
|:-------------------------:|:-------------------------:|
|<img width="450" alt="screen" src="demo/italia_before.png"> | <img width="450" alt="screen" src="demo/italia_after.png"> |

## Evaluation Metrics
  
| Evaluate anonymization using information loss metrics |
|:-------------------------:|
|<img width="1000" height="600" alt="screen" src="demo/metrics.png"> |
|<img width="1000" height="600" alt="screen" src="demo/metrics2.png"> |
  

| Evaluate anonymization using classification models |
|:-------------------------:|
|<img width="1000" height="600" alt="screen" src="demo/metrics_ml.png"> |
|<img width="1000" height="600" alt="screen" src="demo/metrics_ml2.png"> |



## References:
- Top-Down Greedy, Cluster-based (https://github.com/fhstp/k-AnonML)
- L-Diversity: Machanavajjhala et al., l-Diversity: Privacy beyond k-Anonymity, ACM TKDD 2007
- Datafly Algorithm (https://github.com/nazilkbahar/python-datafly)
- Normalized Certainty Penalty from [Utility-Based Anonymization for Privacy Preservation with
Less Information Loss](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.450.6140&rep=rep1&type=pdf)
- Discernibility, Average Equivalent Class Size from [A Systematic Comparison and Evaluation
of k-Anonymization Algorithms
for Practitioners](http://www.tdp.cat/issues11/tdp.a169a14.pdf)
- [Privacy in a Mobile-Social World](https://courses.cs.duke.edu//fall12/compsci590.3/slides/lec3.pdf)
- Code and idea based on [k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers](https://arxiv.org/abs/2102.04763)

## Citation
```
Minh-Khoi Pham. (2026). kaylode/k-anonymity: v1 (Version v1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22748400
```
