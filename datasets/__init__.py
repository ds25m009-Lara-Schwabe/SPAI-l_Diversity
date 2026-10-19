
from utils.types import Dataset

def get_dataset_params(name):
    if name == Dataset.ADULT:
        # occupation is the sensitive attribute
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7]
        target_var = 'salary-class'
        default_sa = 'occupation'
        IS_CAT = [True, False, True, True, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.CMC:
        QI_INDEX = [1, 2, 4]
        target_var = 'method'
        default_sa = 'method'
        IS_CAT = [False, True, False]
        max_numeric = {"age": 32.5, "children": 8}
    elif name == Dataset.MGM:
        QI_INDEX = [1, 2, 3, 4, 5]
        target_var = 'severity'
        default_sa = 'severity'
        IS_CAT = [True, False, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.CAHOUSING:
        QI_INDEX = [1, 2, 3, 8, 9]
        target_var = 'ocean_proximity'
        default_sa = 'ocean_proximity'
        IS_CAT = [False, False, False, False, False]
        max_numeric = {"latitude": 119.33, "longitude": 37.245, "housing_median_age": 32.5,
                    "median_house_value": 257500, "median_income": 5.2035}
    elif name == Dataset.INFORMS:
        QI_INDEX = [3, 4, 6, 13, 16]
        target_var = "poverty"
        default_sa = "poverty"
        IS_CAT = [True, True, True, True, False]
        max_numeric = {"DOBMM": None, "DOBYY": None, "RACEX":None, "EDUCYEAR": None, "income": None}
    elif name == Dataset.ITALIA:
        QI_INDEX = [1, 2, 3]
        target_var = "disease"
        default_sa = "disease"
        IS_CAT = [False, True, False]
        max_numeric = {"age": 50, "city_birth": None, "zip_code":50000}
    else:
        print(f"Not support {name} dataset")
        raise ValueError
    return {
        'qi_index': QI_INDEX,
        'is_category': IS_CAT,
        'target_var': target_var,
        'sa_var': default_sa,
        'max_numeric': max_numeric
    }


def get_sa_qi_params(name, att_names, sa_var=None):
    """
    qi_index and is_category without the sensitive attribute
    """
    params = get_dataset_params(name)
    sa_var = sa_var or params['sa_var']
    if sa_var not in att_names:
        raise ValueError(f"{sa_var} is not a column of {name}")
    sa_col = att_names.index(sa_var)
    qi_index, is_cat = [], []
    for qi, cat in zip(params['qi_index'], params['is_category']):
        if qi != sa_col:
            qi_index.append(qi)
            is_cat.append(cat)
    return qi_index, is_cat, sa_var