from utils.types import AnonMethod
import os
import argparse
import numpy as np
import pandas as pd
from metrics import NCP, DM, CAVG, LDiversity

from algorithms import (
        k_anonymize,
        read_tree)
from datasets import get_sa_qi_params
from utils.data import read_raw, write_anon

parser = argparse.ArgumentParser('K-Anonymize')
parser.add_argument('--method', type=str, default='topdown',
                    choices=['topdown', 'topdown_ldiv', 'cluster', 'datafly'],
                    help="K-Anonymity Method")
parser.add_argument('--k', type=int, default=2,
                    help="K-Anonymity")
parser.add_argument('--l', type=int, default=0,
                    help="L-Diversity (topdown_ldiv)")
parser.add_argument('--sa', type=str, default=None,
                    help="Sensitive attribute")
parser.add_argument('--ldiv', type=str, default='distinct',
                    choices=['distinct', 'entropy', 'recursive'],
                    help="L-Diversity type")
parser.add_argument('--c', type=float, default=1.0,
                    help="c for recursive (c,l)-diversity")
parser.add_argument('--dataset', type=str, default='adult',
                    help="Dataset to anonymize")

class Anonymizer:
    def __init__(self, args):
        self.method = args.method
        assert self.method in ["topdown", "topdown_ldiv", "cluster", "datafly"]
        self.k = args.k
        self.l = getattr(args, 'l', 0) or 0
        self.sa = getattr(args, 'sa', None)
        self.ldiv_type = getattr(args, 'ldiv', 'distinct')
        self.c = getattr(args, 'c', 1.0)
        if self.method == AnonMethod.TOPDOWN_LDIV and self.l < 1:
            raise ValueError("topdown_ldiv needs --l")
        self.data_name = args.dataset
        self.csv_path = args.dataset+'.csv'

        # Data path
        self.path = os.path.join('data', args.dataset)  # trailing /

        # Dataset path
        self.data_path = os.path.join(self.path, self.csv_path)

        # Generalization hierarchies path
        self.gen_path = os.path.join(
            self.path,
            'hierarchies')  # trailing /

        # folder for all results
        res_folder = os.path.join(
            'results', 
            args.dataset, 
            self.method)

        # path for anonymized datasets
        self.anon_folder = res_folder  # trailing /
        
        os.makedirs(self.anon_folder, exist_ok=True)

    def anonymize(self):
        data = pd.read_csv(self.data_path, delimiter=';')
        ATT_NAMES = list(data.columns)
        
        QI_INDEX, _, SA_var = get_sa_qi_params(
            self.data_name, ATT_NAMES, self.sa)
        SA_COL = ATT_NAMES.index(SA_var)

        QI_NAMES = list(np.array(ATT_NAMES)[QI_INDEX])
        IS_CAT = [True] * len(QI_INDEX) # is all cat because all hierarchies are provided
        SA_INDEX = [index for index in range(len(ATT_NAMES)) if index not in QI_INDEX]

        ATT_TREES = read_tree(
            self.gen_path, 
            self.data_name, 
            ATT_NAMES, 
            QI_INDEX, IS_CAT)

        raw_data, header = read_raw(
            self.path, 
            self.data_name, 
            QI_INDEX, IS_CAT)

        anon_params = {
            "name" :self.method,
            "att_trees" :ATT_TREES,
            "value" :self.k,
            "qi_index" :QI_INDEX, 
            "sa_index" :SA_INDEX,
            "l": self.l,
            "sa_col": SA_COL,
            "ldiv_type": self.ldiv_type,
            "c": self.c
        }

        if self.method == AnonMethod.DATAFLY:
            anon_params.update({
                'qi_names': QI_NAMES,
                'csv_path': self.data_path,
                'data_name': self.data_name,
                'dgh_folder': self.gen_path,
                'res_folder': self.anon_folder})

        anon_params.update({'data': raw_data})

        print(f"Anonymize with {self.method}")
        anon_data, runtime = k_anonymize(anon_params)

        # Write anonymized table
        if anon_data is not None:
            nodes_count = write_anon(
                self.anon_folder, 
                anon_data, 
                header, 
                self.k, 
                self.data_name,
                l=self.l if self.method == AnonMethod.TOPDOWN_LDIV else None,
                ldiv_type=self.ldiv_type)

        # Normalized Certainty Penalty
        ncp = NCP(anon_data, QI_INDEX, ATT_TREES)
        ncp_score = ncp.compute_score()

        # Discernibility Metric

        raw_dm = DM(raw_data, QI_INDEX, self.k)
        raw_dm_score = raw_dm.compute_score()

        anon_dm = DM(anon_data, QI_INDEX, self.k)
        anon_dm_score = anon_dm.compute_score()

        # Average Equivalence Class

        raw_cavg = CAVG(raw_data, QI_INDEX, self.k)
        raw_cavg_score = raw_cavg.compute_score()

        anon_cavg = CAVG(anon_data, QI_INDEX, self.k)
        anon_cavg_score = anon_cavg.compute_score()

        print(f"NCP score (lower is better): {ncp_score:.3f}")
        print(f"CAVG score (near 1 is better): BEFORE: {raw_cavg_score:.3f} || AFTER: {anon_cavg_score:.3f}")
        print(f"DM score (lower is better): BEFORE: {raw_dm_score} || AFTER: {anon_dm_score}")
        print(f"Time execution: {runtime:.3f}s")

        # L-Diversity
        ldiv = LDiversity(anon_data, QI_INDEX, SA_COL, self.k, max(self.l, 1),
                          self.ldiv_type, self.c)
        ldiv_result = ldiv.compute_score()
        print(f"L-Diversity ({SA_var}): {ldiv_result['num_eqs']} ECs, "
              f"min distinct l: {ldiv_result['min_distinct_l']}, "
              f"min entropy l: {ldiv_result['min_entropy_l']:.2f}, "
              f"homogeneous ECs: {ldiv_result['homogeneous_eqs']}")
        if self.l > 0:
            print(f"ECs violating l={self.l}: {ldiv_result['violating_eqs']}")

        self.scores = {
            'ncp': ncp_score,
            'cavg': anon_cavg_score,
            'dm': anon_dm_score,
            'runtime': runtime,
            **ldiv_result
        }

        return ncp_score, raw_cavg_score, anon_cavg_score, raw_dm_score, anon_dm_score


def main(args):
    try:
        anonymizer = Anonymizer(args)
        anonymizer.anonymize()
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)