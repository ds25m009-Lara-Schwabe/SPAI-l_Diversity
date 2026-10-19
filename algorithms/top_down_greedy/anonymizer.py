# -*- coding: utf-8 -*-
"""
run top_down_greedy_anonymization with given argv
"""

import copy
import sys
import os

from .top_down_greedy_anonymization import \
    Top_Down_Greedy_Anonymization
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import reorder_columns, restore_column_order



def tdg_anonymize(k, att_trees, data, qi_index, sa_index, **kwargs):
    """
    Top-Down Greedy Anonymization
    """

    result, runtime = Top_Down_Greedy_Anonymization(att_trees, reorder_columns(
        copy.deepcopy(data), qi_index), k, len(qi_index), sa_index)
    

    return restore_column_order(result, qi_index), runtime


def tdg_ldiv_anonymize(k, l, att_trees, data, qi_index, sa_index, sa_col,
                       ldiv_type='distinct', c=1.0, **kwargs):
    sa_pos = len(qi_index) + sa_index.index(sa_col)
    result, runtime = Top_Down_Greedy_Anonymization(
        att_trees, reorder_columns(copy.deepcopy(data), qi_index), k,
        len(qi_index), sa_index, l=l, sa_pos=sa_pos, ldiv_type=ldiv_type, c=c)
    return restore_column_order(result, qi_index), runtime
