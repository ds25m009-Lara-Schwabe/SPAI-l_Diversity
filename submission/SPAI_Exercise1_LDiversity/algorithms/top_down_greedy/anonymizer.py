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


# l-diversity extension
def tdg_anonymize(k, att_trees, data, qi_index, sa_index, l=None, sensitive_index=None, **kwargs):
    """
    Top-Down Greedy Anonymization
    """
    #l-diversity requires a defined sensitive atribute
    if l is not None and sensitive_index is None:
        raise ValueError("sensitive_index must be provided when l-diversity is enabled")
    
    # moving all QI to the beginning of each record
    # as expected by the TDG implementation
    reordered_data = reorder_columns(
        copy.deepcopy(data),
        qi_index
    )

    # converting the original sensitive-attribute index to its position after reorder_columns()
    if sensitive_index is not None:
        reordered_indices = qi_index + [
            i for i in range(len(data[0])) if i not in qi_index
        ]
        sensitive_index = reordered_indices.index(sensitive_index)


    result, runtime = Top_Down_Greedy_Anonymization(
        att_trees,
        reordered_data,
        k,
        len(qi_index),
        sa_index,
        l=l,
        sensitive_index=sensitive_index
    )
    

    return restore_column_order(result, qi_index), runtime
