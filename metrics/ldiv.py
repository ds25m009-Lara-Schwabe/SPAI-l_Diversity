import math
from algorithms.top_down_greedy.top_down_greedy_anonymization import is_l_diverse


class LDiversity:
    """
    L-Diversity of an anonymized table based on definition from
    https://dl.acm.org/doi/10.1145/1217299.1217302
    """
    def __init__(self, anon_data, qi_index, sa_col, k=1, l=1, ldiv_type='distinct', c=1.0) -> None:
        self.anon_data = anon_data
        self.qi_index = qi_index
        self.sa_col = sa_col
        self.k = k
        self.l = l
        self.ldiv_type = ldiv_type
        self.c = c

    def compute_eq(self):
        self.eq_count = {}
        for record in self.anon_data:
            eq = tuple(record[qi_id] for qi_id in self.qi_index)
            sa_value = record[self.sa_col]
            if eq not in self.eq_count.keys():
                self.eq_count[eq] = {}
            self.eq_count[eq][sa_value] = self.eq_count[eq].get(sa_value, 0) + 1

    def entropy_l(self, counts):
        total = sum(counts)
        entropy = -sum(n / total * math.log(n / total) for n in counts)
        return math.exp(entropy)

    def compute_score(self):
        self.compute_eq()
        result = {
            'num_eqs': len(self.eq_count),
            'min_eq_size': len(self.anon_data),
            'min_distinct_l': len(self.anon_data),
            'min_entropy_l': float(len(self.anon_data)),
            'homogeneous_eqs': 0,
            'homogeneous_records': 0,
            'violating_eqs': 0,
            'violating_records': 0,
            'majority_share': 0.0,
        }
        for sa_count in self.eq_count.values():
            counts = list(sa_count.values())
            size = sum(counts)
            result['min_eq_size'] = min(result['min_eq_size'], size)
            result['min_distinct_l'] = min(result['min_distinct_l'], len(counts))
            result['min_entropy_l'] = min(result['min_entropy_l'], self.entropy_l(counts))
            result['majority_share'] += max(counts) / len(self.anon_data)
            if len(counts) == 1:
                result['homogeneous_eqs'] += 1
                result['homogeneous_records'] += size
            if size < self.k or not is_l_diverse(counts, self.l, self.ldiv_type, self.c):
                result['violating_eqs'] += 1
                result['violating_records'] += size
        result['satisfied'] = result['violating_eqs'] == 0
        return result
