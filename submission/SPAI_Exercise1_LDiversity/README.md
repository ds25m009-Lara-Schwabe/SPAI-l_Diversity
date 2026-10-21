# SPAI Exercise 1 – l-Diversity

Implementation of distinct l-diversity on top of the existing
Top-down Greedy k-anonymity algorithm.

## Dataset
cahousing

## Sensitive attribute
ocean_proximity

## Run

k-anonymity:

python anonymize.py --method=topdown --k=5 --dataset=cahousing

k-anonymity + l-diversity:

python anonymize.py --method=topdown --k=5 --l=2 --dataset=cahousing

## Evaluation

See `Evaluation/Evaluation.ipynb`.

Tested configurations:
- k = 2, 5, 10
- l = 2, 3

## Authors
Lara Schwabe
Livia Zylja
