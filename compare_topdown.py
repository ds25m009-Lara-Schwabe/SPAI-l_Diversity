import os
import io
import hashlib
import argparse
import contextlib
from argparse import Namespace

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import algorithms.top_down_greedy.top_down_greedy_anonymization as tdg
from anonymize import Anonymizer
from models import classifier_evaluation
from datasets import get_sa_qi_params
from algorithms import read_tree
from utils.data import anon_filename


def int_list(value):
    return [int(v) for v in value.split(',') if v]


def variant_list(value):
    return [(v.split(':')[0], int(v.split(':')[1])) for v in value.split(',') if v]


parser = argparse.ArgumentParser('Top-Down Greedy vs Top-Down Greedy with l-diversity')
parser.add_argument('--dataset', type=str, default='cahousing')
parser.add_argument('--sa', type=str, default=None)
parser.add_argument('--k_values', type=int_list, default=[2, 3, 5, 10, 20, 50, 100])
parser.add_argument('--variants', type=variant_list, default=[('distinct', 2), ('distinct', 3), ('entropy', 2)])
parser.add_argument('--baseline_l', type=int, default=3)
parser.add_argument('--strategy_k', type=int_list, default=[2, 10])
parser.add_argument('--strategy_l', type=int_list, default=[2, 3])
parser.add_argument('--k_fixed', type=int, default=10)
parser.add_argument('--recursive_l', type=int_list, default=[2, 3])
parser.add_argument('--c', type=float, default=2.0)
parser.add_argument('--classifiers', type=str, default='knn,svm,rf')
parser.add_argument('--out', type=str, default='experiments')
parser.add_argument('--plot_only', action='store_true')

ml_names = {'knn': 'KNN', 'svm': 'SVM', 'rf': 'Random Forest'}
strategies = ['reject', 'retry', 'repair']


def reject_split(sub_partitions):
    if all(tdg.is_valid(tdg.sa_count(p.member)) for p in sub_partitions):
        return sub_partitions
    return None


def retry_split(sub_partitions, attempts=10):
    if all(tdg.is_valid(tdg.sa_count(p.member)) for p in sub_partitions):
        return sub_partitions
    members = sub_partitions[0].member + sub_partitions[1].member
    parent = tdg.Partition(members, tdg.middle_group(members))
    for _ in range(attempts - 1):
        u, v = tdg.get_pair(parent)
        subs = tdg.distribute_record(u, v, parent)
        if len(subs[0]) < tdg.GL_K:
            tdg.balance(subs, 0)
        elif len(subs[1]) < tdg.GL_K:
            tdg.balance(subs, 1)
        if len(subs) == 2 and all(tdg.is_valid(tdg.sa_count(p.member)) for p in subs):
            return subs
    return None


def anonymize(method, dataset, k, l, sa, ldiv_type, c, strategy='repair'):
    anonymizer = Anonymizer(Namespace(method=method, dataset=dataset, k=k, l=l, sa=sa, ldiv=ldiv_type, c=c))
    if strategy != 'repair':
        anonymizer.anon_folder += '_' + strategy
        os.makedirs(anonymizer.anon_folder, exist_ok=True)
    balance_l = tdg.balance_l
    if strategy == 'reject':
        tdg.balance_l = reject_split
    elif strategy == 'retry':
        tdg.balance_l = retry_split
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            anonymizer.anonymize()
    finally:
        tdg.balance_l = balance_l
    if method == 'topdown_ldiv':
        name = anon_filename(dataset, k, l, ldiv_type)
    else:
        name = anon_filename(dataset, k)
    return anonymizer.scores, os.path.join(anonymizer.anon_folder, name)


class Evaluator:
    def __init__(self, dataset, sa, classifiers):
        self.classifiers = classifiers
        path = os.path.join('data', dataset)
        self.raw_csv = os.path.join(path, f'{dataset}.csv')
        self.train_index = os.path.join(path, f'{dataset}_train.txt')
        self.val_index = os.path.join(path, f'{dataset}_val.txt')
        att_names = list(pd.read_csv(self.raw_csv, delimiter=';', nrows=1).columns)
        self.qi_index, self.is_cat, _ = get_sa_qi_params(dataset, att_names, sa)
        self.att_trees = read_tree(os.path.join(path, 'hierarchies'), dataset, att_names,
                                   self.qi_index, [True] * len(self.qi_index))
        self.cache = {}

    def evaluate(self, anon_csv=None):
        key = None
        if anon_csv is not None:
            with open(anon_csv, 'rb') as f:
                key = hashlib.md5(f.read()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        scores = {}
        for name in self.classifiers:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if anon_csv is None:
                    f1 = classifier_evaluation(name, self.raw_csv, self.train_index, self.val_index,
                                               self.qi_index, self.is_cat)
                else:
                    f1 = classifier_evaluation(name, self.raw_csv, self.train_index, self.val_index,
                                               anon_csv=anon_csv, qi_index=self.qi_index,
                                               is_cat=self.is_cat, att_trees=self.att_trees)
            scores[f'f1_{name}'] = f1
        self.cache[key] = scores
        return scores


def experiment_runs(args):
    runs = []
    for k in args.k_values:
        runs.append(('B', 'topdown', '-', k, args.baseline_l, 'repair'))
        runs += [('B', 'topdown_ldiv', t, k, l, 'repair') for t, l in args.variants]
    for k in args.strategy_k:
        for l in args.strategy_l:
            runs += [('C', 'topdown_ldiv', 'distinct', k, l, s) for s in strategies]
    runs += [('R', 'topdown_ldiv', 'recursive', args.k_fixed, l, 'repair') for l in args.recursive_l]
    return runs


def run_experiments(args, out_csv):
    classifiers = [c for c in args.classifiers.split(',') if c]
    evaluator = Evaluator(args.dataset, args.sa, classifiers) if classifiers else None

    rows = []
    if evaluator is not None:
        rows.append({'experiment': 'original', 'method': 'original', 'ldiv_type': '-',
                     'k': 1, 'l': 0, 'strategy': '-', **evaluator.evaluate()})

    done = {}
    runs = experiment_runs(args)
    for i, (exp, method, ldiv_type, k, l, strategy) in enumerate(runs):
        key = (method, ldiv_type, k, l, strategy)
        print(f'[{i + 1}/{len(runs)}] {exp} {method} {ldiv_type} k={k} l={l} {strategy}', flush=True)
        if key not in done:
            try:
                scores, csv_path = anonymize(method, args.dataset, k, l, args.sa,
                                             ldiv_type if method == 'topdown_ldiv' else 'distinct', args.c, strategy)
            except ValueError as e:
                print('skipped:', e)
                continue
            if evaluator is not None:
                scores.update(evaluator.evaluate(csv_path))
            done[key] = scores
        row = {'experiment': exp, 'method': method, 'ldiv_type': ldiv_type,
               'k': k, 'l': l, 'strategy': strategy, **done[key]}
        print(f"  ncp={row['ncp']:.3f} ecs={row['num_eqs']} homogeneous={row['homogeneous_eqs']} "
              f"violating={row['violating_eqs']} time={row['runtime']:.1f}s "
              + ' '.join(f"{c}={row[f'f1_{c}']:.3f}" for c in classifiers), flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows)


def label(method, ldiv_type, l):
    if method == 'topdown':
        return 'Top-Down Greedy (k-anonymity)'
    return f'TDG + {ldiv_type} l = {l}'


def plot(df, metrics, titles, figname, logy=(), baseline=None, ylim=None):
    fig, axis = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    ticks = sorted(df['k'].unique())
    for ax, metric, title in zip(axis, metrics, titles):
        for (method, ldiv_type, l), sub in df.groupby(['method', 'ldiv_type', 'l'], sort=False):
            sub = sub.sort_values('k')
            ax.plot(sub['k'], sub[metric], marker='o', label=label(method, ldiv_type, l))
        if baseline is not None:
            ax.axhline(baseline[metric], color='grey', linestyle='--', label='original data')
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        if metric in logy:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel('k')
        ax.grid(alpha=0.3)
    handles, labels = axis[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figname, dpi=200)
    plt.close(fig)
    print('saved', figname)


def plot_strategies(exp_c, figname):
    settings = exp_c[['k', 'l']].drop_duplicates().sort_values(['k', 'l']).values.tolist()
    fig, axis = plt.subplots(1, 2, figsize=(9, 3.4))
    width = 0.27
    for ax, metric, title in zip(axis, ['ncp', 'num_eqs'], ['NCP', 'number of ECs']):
        for j, s in enumerate(strategies):
            values = [exp_c[(exp_c.k == k) & (exp_c.l == l) & (exp_c.strategy == s)][metric].iloc[0]
                      for k, l in settings]
            ax.bar([i + (j - 1) * width for i in range(len(settings))], values, width, label=f'split {s}ed'
                   if s != 'retry' else 'split retried (10x)')
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([f'k = {k}\nl = {l}' for k, l in settings])
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='y')
    handles, labels = axis[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figname, dpi=200)
    plt.close(fig)
    print('saved', figname)


def plot_results(df, dataset, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    clfs = [c for c in df.columns if c.startswith('f1_')]
    original = df[df['method'] == 'original']
    baseline = original.iloc[0] if len(original) else None
    exp_b = df[df['experiment'] == 'B']
    exp_c = df[df['experiment'] == 'C']
    fig = lambda name: os.path.join(fig_dir, f'{dataset}_{name}.png')
    n_records = len(pd.read_csv(os.path.join('data', dataset, f'{dataset}.csv'), delimiter=';', usecols=[0]))

    if len(exp_b):
        plot(exp_b, ['ncp', 'num_eqs', 'cavg'], ['NCP', 'number of ECs', 'CAVG'], fig('info_vs_k'),
             logy=('num_eqs',))
        priv = exp_b.assign(homogeneous_share=100 * exp_b['homogeneous_records'] / n_records,
                            majority_pct=100 * exp_b['majority_share'])
        plot(priv, ['homogeneous_eqs', 'homogeneous_share', 'majority_pct'],
             ['homogeneous ECs (one SA value)', 'records in homogeneous ECs [%]',
              'records with majority SA value of EC [%]'], fig('privacy_vs_k'))
        if clfs:
            ml_titles = [ml_names[c[3:]] + ' (weighted F1)' for c in clfs]
            low = df[clfs].min().min()
            high = max(df[clfs].max().max(), baseline[clfs].max() if baseline is not None else 0)
            plot(exp_b, clfs, ml_titles, fig('ml_vs_k'), baseline=baseline, ylim=(low - 0.03, high + 0.03))
    if len(exp_c):
        plot_strategies(exp_c, fig('strategies'))


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f'{args.dataset}_topdown_ldiv_results.csv')
    if args.plot_only:
        df = pd.read_csv(out_csv)
    else:
        df = run_experiments(args, out_csv)
    plot_results(df, args.dataset, os.path.join(args.out, 'figures'))
