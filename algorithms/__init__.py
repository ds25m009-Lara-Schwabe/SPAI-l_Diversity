from algorithms.datafly import datafly_anonymize
from .utils.read_data import read_tree
from .clustering_based import cluster_based_anonymize
from .top_down_greedy import tdg_anonymize, tdg_ldiv_anonymize
from utils.types import AnonMethod

def k_anonymize(anon_params):

    if anon_params["name"] == AnonMethod.CLUSTER:
        return cluster_based_anonymize(
            anon_params["value"], 
            anon_params["att_trees"], 
            anon_params["data"], 
            anon_params["qi_index"], 
            anon_params["sa_index"], 
            type_alg='kmember')

    if anon_params["name"] == AnonMethod.TOPDOWN:
        return tdg_anonymize(
            anon_params["value"], 
            anon_params["att_trees"], 
            anon_params["data"], 
            anon_params["qi_index"], 
            anon_params["sa_index"])

    if anon_params["name"] == AnonMethod.TOPDOWN_LDIV:
        return tdg_ldiv_anonymize(
            anon_params["value"],
            anon_params["l"],
            anon_params["att_trees"],
            anon_params["data"],
            anon_params["qi_index"],
            anon_params["sa_index"],
            anon_params["sa_col"],
            ldiv_type=anon_params["ldiv_type"],
            c=anon_params["c"])

    if anon_params["name"] == AnonMethod.DATAFLY:
        return datafly_anonymize(
            anon_params["value"], 
            anon_params["csv_path"], 
            anon_params["qi_names"], 
            anon_params["data_name"], 
            anon_params["dgh_folder"],
            anon_params['res_folder'])