import os
import io
import argparse
import contextlib
from argparse import Namespace
from collections import Counter

import pandas as pd

from anonymize import Anonymizer
from datasets import get_sa_qi_params
from utils.data import anon_filename

parser = argparse.ArgumentParser('Top-Down Greedy l-diversity demo')
parser.add_argument('--dataset', type=str, default='italia')
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--l', type=int, default=3)
parser.add_argument('--sa', type=str, default=None)
parser.add_argument('--out', type=str, default='experiments')


def equivalence_classes(csv_path, qi_names, sa):
    df = pd.read_csv(csv_path, delimiter=';', dtype=str)
    rows = []
    for qi_values, group in df.groupby(qi_names, sort=False):
        counts = Counter(group[sa])
        row = dict(zip(qi_names, qi_values))
        row['size'] = len(group)
        row['distinct'] = len(counts)
        row[sa] = ', '.join(f'{v} ({n})' for v, n in counts.most_common())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['distinct', 'size']).reset_index(drop=True)


def main(args):
    raw_csv = os.path.join('data', args.dataset, f'{args.dataset}.csv')
    att_names = list(pd.read_csv(raw_csv, delimiter=';', nrows=1).columns)
    qi_index, _, sa = get_sa_qi_params(args.dataset, att_names, args.sa)
    qi_names = [att_names[i] for i in qi_index]

    lines = [f'# {args.dataset}, k = {args.k}, l = {args.l}, sensitive attribute: {sa}', '']
    for method in ['topdown', 'topdown_ldiv']:
        anonymizer = Anonymizer(Namespace(method=method, dataset=args.dataset, k=args.k, l=args.l,
                                          sa=sa, ldiv='distinct', c=1.0))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            anonymizer.anonymize()
        scores = anonymizer.scores
        l = args.l if method == 'topdown_ldiv' else None
        ecs = equivalence_classes(os.path.join(anonymizer.anon_folder, anon_filename(args.dataset, args.k, l)),
                                  qi_names, sa)
        title = (f"{method}: {scores['num_eqs']} ECs, NCP = {scores['ncp']:.3f}, "
                 f"homogeneous ECs = {scores['homogeneous_eqs']}, "
                 f"ECs violating l = {args.l}: {scores['violating_eqs']}")
        print('\n' + title)
        print(ecs.to_string())
        lines += ['## ' + title, '', '```', ecs.to_string(index=False), '```', '']

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f'{args.dataset}_demo_k{args.k}_l{args.l}.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main(parser.parse_args())
