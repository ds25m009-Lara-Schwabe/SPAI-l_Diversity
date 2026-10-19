import os
import io
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


parser = argparse.ArgumentParser('Top-Down Greedy vs Top-Down Greedy with l-diversity')
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--sa', type=str, default=None)
parser.add_argument('--k_fixed', type=int, default=10)
parser.add_argument('--l_values', type=int_list, default=[2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--entropy_l_values', type=int_list, default=[2, 3, 4, 5, 6])
parser.add_argument('--recursive_l_values', type=int_list, default=[2, 3, 4])
parser.add_argument('--c', type=float, default=2.0)
parser.add_argument('--l_fixed', type=int, default=3)
parser.add_argument('--k_values', type=int_list, default=[2, 3, 5, 10, 20, 50, 100])
parser.add_argument('--classifiers', type=str, default='knn,svm,rf')
parser.add_argument('--out', type=str, default='experiments')
parser.add_argument('--plot_only', action='store_true')

ml_names = {'knn': 'KNN', 'svm': 'SVM', 'rf': 'Random Forest'}


def no_moving(sub_partitions):
    if all(tdg.is_valid(tdg.sa_count(p.member)) for p in sub_partitions):
        return sub_partitions
    return None


def anonymize(method, dataset, k, l, sa, ldiv_type, c, naive=False):
    anonymizer = Anonymizer(Namespace(method=method, dataset=dataset, k=k, l=l, sa=sa, ldiv=ldiv_type, c=c))
    if naive:
        anonymizer.anon_folder += '_naive'
        os.makedirs(anonymizer.anon_folder, exist_ok=True)
    balance_l = tdg.balance_l
    if naive:
        tdg.balance_l = no_moving
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

    def evaluate(self, anon_csv=None):
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
        return scores


def experiment_runs(args):
    runs = [('A', 'topdown', '-', args.k_fixed, args.l_fixed, False)]
    runs += [('A', 'topdown_ldiv', 'distinct', args.k_fixed, l, False) for l in args.l_values]
    runs += [('A', 'topdown_ldiv', 'entropy', args.k_fixed, l, False) for l in args.entropy_l_values]
    runs += [('A', 'topdown_ldiv', 'recursive', args.k_fixed, l, False) for l in args.recursive_l_values]
    for k in args.k_values:
        runs.append(('B', 'topdown', '-', k, args.l_fixed, False))
        runs.append(('B', 'topdown_ldiv', 'distinct', k, args.l_fixed, False))
    runs += [('C', 'topdown_ldiv', 'distinct', args.k_fixed, l, True) for l in [args.l_fixed, 6]]
    return runs


def run_experiments(args, out_csv):
    classifiers = [c for c in args.classifiers.split(',') if c]
    evaluator = Evaluator(args.dataset, args.sa, classifiers) if classifiers else None

    rows = []
    if evaluator is not None:
        rows.append({'experiment': 'original', 'method': 'original', 'ldiv_type': '-',
                     'k': 1, 'l': 0, 'naive': False, **evaluator.evaluate()})

    done = {}
    runs = experiment_runs(args)
    for i, (exp, method, ldiv_type, k, l, naive) in enumerate(runs):
        key = (method, ldiv_type, k, l, naive)
        print(f'[{i + 1}/{len(runs)}] {exp} {method} {ldiv_type} k={k} l={l} naive={naive}', flush=True)
        if key not in done:
            try:
                scores, csv_path = anonymize(method, args.dataset, k, l, args.sa,
                                             ldiv_type if method == 'topdown_ldiv' else 'distinct', args.c, naive)
            except ValueError as e:
                print('skipped:', e)
                continue
            if evaluator is not None:
                scores.update(evaluator.evaluate(csv_path))
            done[key] = scores
        row = {'experiment': exp, 'method': method, 'ldiv_type': ldiv_type,
               'k': k, 'l': l, 'naive': naive, **done[key]}
        print(f"  ncp={row['ncp']:.3f} ecs={row['num_eqs']} homogeneous={row['homogeneous_eqs']} "
              f"violating={row['violating_eqs']} time={row['runtime']:.1f}s "
              + ' '.join(f"{c}={row[f'f1_{c}']:.3f}" for c in classifiers), flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows)


def label(method, ldiv_type):
    if method == 'topdown':
        return 'Top-Down Greedy (k-anonymity)'
    return f'TDG + {ldiv_type} l-diversity'


def plot(df, x, metrics, titles, figname, logx=False, logy=(), baseline=None, ylim=None):
    fig, axis = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    for ax, metric, title in zip(axis, metrics, titles):
        for (method, ldiv_type), sub in df.groupby(['method', 'ldiv_type'], sort=False):
            sub = sub.sort_values(x)
            if len(sub) == 1:
                ax.axhline(sub[metric].iloc[0], color='black', linestyle=':', label=label(method, ldiv_type))
            else:
                ax.plot(sub[x], sub[metric], marker='o', label=label(method, ldiv_type))
        if baseline is not None:
            ax.axhline(baseline[metric], color='grey', linestyle='--', label='original data')
        ticks = sorted(df[x].unique())
        if logx:
            ax.set_xscale('log')
            ax.minorticks_off()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        if metric in logy:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.grid(alpha=0.3)
    axis[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(figname, dpi=200)
    plt.close(fig)
    print('saved', figname)


def plot_results(df, dataset, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    clfs = [c for c in df.columns if c.startswith('f1_')]
    original = df[df['method'] == 'original']
    baseline = original.iloc[0] if len(original) else None
    exp_a = df[(df['experiment'] == 'A') & (df['ldiv_type'] != 'recursive')]
    exp_b = df[df['experiment'] == 'B']
    fig = lambda name: os.path.join(fig_dir, f'{dataset}_{name}.png')
    ml_titles = [ml_names[c[3:]] + ' (weighted F1)' for c in clfs]
    f1_range = (df[clfs].min().min() - 0.02, df[clfs].max().max() + 0.02) if clfs else None

    if len(exp_a):
        plot(exp_a, 'l', ['ncp', 'num_eqs', 'cavg'], ['NCP', 'number of ECs', 'CAVG'], fig('info_vs_l'))
        if clfs:
            plot(exp_a, 'l', clfs, ml_titles, fig('ml_vs_l'), baseline=baseline, ylim=f1_range)
    if len(exp_b):
        l = int(exp_b['l'].iloc[0])
        plot(exp_b, 'k', ['ncp', 'dm', 'cavg'], ['NCP', 'Discernibility (DM)', 'CAVG'],
             fig('info_vs_k'), logx=True, logy=('dm',))
        plot(exp_b, 'k', ['homogeneous_eqs', 'violating_eqs', 'min_distinct_l'],
             ['homogeneous ECs', f'ECs violating {l}-diversity', 'min. distinct l'],
             fig('privacy_vs_k'), logx=True)
        if clfs:
            plot(exp_b, 'k', clfs, ml_titles, fig('ml_vs_k'), logx=True, baseline=baseline, ylim=f1_range)
    if len(exp_a) and len(exp_b):
        plot_runtime(exp_a, exp_b, fig('runtime'))
    exp_c = df[df['experiment'] == 'C']
    if len(exp_c):
        plot_repair(df, exp_c, fig('repair'))


def plot_runtime(exp_a, exp_b, figname):
    fig, axis = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, sub_df, x in [(axis[0], exp_a, 'l'), (axis[1], exp_b, 'k')]:
        for (method, ldiv_type), sub in sub_df.groupby(['method', 'ldiv_type'], sort=False):
            sub = sub.sort_values(x)
            if len(sub) == 1:
                ax.axhline(sub['runtime'].iloc[0], color='black', linestyle=':', label=label(method, ldiv_type))
            else:
                ax.plot(sub[x], sub['runtime'], marker='o', label=label(method, ldiv_type))
        ax.set_xlabel(x)
        ax.set_ylabel('runtime [s]')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axis[1].set_xscale('log')
    axis[1].minorticks_off()
    ticks = sorted(exp_b['k'].unique())
    axis[1].set_xticks(ticks)
    axis[1].set_xticklabels(ticks)
    axis[0].set_title(f"k = {exp_a['k'].iloc[0]}")
    axis[1].set_title(f"l = {exp_b['l'].iloc[0]}")
    axis[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figname, dpi=200)
    plt.close(fig)
    print('saved', figname)


def plot_repair(df, exp_c, figname):
    k = exp_c['k'].iloc[0]
    repaired = df[(df['experiment'] == 'A') & (df['method'] == 'topdown_ldiv') &
                  (df['ldiv_type'] == 'distinct') & (df['k'] == k)].set_index('l')
    ls = sorted(set(exp_c['l']) & set(repaired.index))
    naive = exp_c.set_index('l')
    fig, axis = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, metric, title in zip(axis, ['ncp', 'num_eqs'], ['NCP', 'number of ECs']):
        x = range(len(ls))
        ax.bar([i - 0.2 for i in x], [naive.loc[l, metric] for l in ls], 0.4, label='split rejected')
        ax.bar([i + 0.2 for i in x], [repaired.loc[l, metric] for l in ls], 0.4, label='split repaired')
        ax.set_xticks(list(x))
        ax.set_xticklabels([f'l = {l}' for l in ls])
        ax.set_title(f'{title} (k = {k})')
        ax.grid(alpha=0.3, axis='y')
    axis[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figname, dpi=200)
    plt.close(fig)
    print('saved', figname)


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f'{args.dataset}_topdown_ldiv_results.csv')
    if args.plot_only:
        df = pd.read_csv(out_csv)
    else:
        df = run_experiments(args, out_csv)
    plot_results(df, args.dataset, os.path.join(args.out, 'figures'))
