import os
import unittest
from collections import Counter

import pandas as pd

from algorithms import read_tree
from algorithms.top_down_greedy import tdg_ldiv_anonymize
from algorithms.top_down_greedy.top_down_greedy_anonymization import is_l_diverse
from datasets import get_sa_qi_params
from metrics import LDiversity
from models.classifiers import replace_generalization
from utils.data import read_raw


class TestIsLDiverse(unittest.TestCase):

    def test_distinct(self):
        self.assertTrue(is_l_diverse([5, 1, 1], 3, 'distinct'))
        self.assertFalse(is_l_diverse([5, 1], 3, 'distinct'))
        self.assertFalse(is_l_diverse([5, 1, 0], 3, 'distinct'))

    def test_entropy(self):
        self.assertTrue(is_l_diverse([2, 2, 2], 3, 'entropy'))
        self.assertFalse(is_l_diverse([8, 1, 1], 3, 'entropy'))

    def test_recursive(self):
        self.assertTrue(is_l_diverse([3, 2, 2], 2, 'recursive', c=1))
        self.assertFalse(is_l_diverse([5, 1, 1], 2, 'recursive', c=1))
        self.assertTrue(is_l_diverse([5, 1, 1], 2, 'recursive', c=3))


class TestLDiversityMetric(unittest.TestCase):

    def setUp(self):
        self.data = [
            ['1', '476**', '2*', 'Gastric Ulcer'],
            ['2', '476**', '2*', 'Gastritis'],
            ['3', '476**', '2*', 'Stomach Cancer'],
            ['4', '4790*', '>=40', 'Flu'],
            ['5', '4790*', '>=40', 'Flu'],
            ['6', '4790*', '>=40', 'Flu'],
        ]

    def test_homogeneous(self):
        result = LDiversity(self.data, [1, 2], 3, k=3, l=3).compute_score()
        self.assertEqual(result['num_eqs'], 2)
        self.assertEqual(result['homogeneous_eqs'], 1)
        self.assertEqual(result['violating_eqs'], 1)
        self.assertFalse(result['satisfied'])
        self.assertAlmostEqual(result['majority_share'], 4 / 6)

    def test_satisfied(self):
        self.data[4][3] = 'Bronchitis'
        self.data[5][3] = 'Pneumonia'
        result = LDiversity(self.data, [1, 2], 3, k=3, l=3).compute_score()
        self.assertTrue(result['satisfied'])
        self.assertAlmostEqual(result['min_entropy_l'], 3.0)


class TestTopDownLDiv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.path.join('data', 'italia')
        att_names = list(pd.read_csv(os.path.join(path, 'italia.csv'), delimiter=';', nrows=1).columns)
        cls.qi_index, _, sa = get_sa_qi_params('italia', att_names)
        cls.sa_col = att_names.index(sa)
        cls.sa_index = [i for i in range(len(att_names)) if i not in cls.qi_index]
        is_cat = [True] * len(cls.qi_index)
        cls.trees = read_tree(os.path.join(path, 'hierarchies'), 'italia', att_names, cls.qi_index, is_cat)
        cls.data, _ = read_raw(path, 'italia', cls.qi_index, is_cat)

    def run_tdg(self, k, l, ldiv_type='distinct', c=1.0):
        result, _ = tdg_ldiv_anonymize(k, l, self.trees, self.data, self.qi_index, self.sa_index,
                                       self.sa_col, ldiv_type=ldiv_type, c=c)
        return result

    def check(self, result, k, l, ldiv_type='distinct', c=1.0):
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(sorted(r[0] for r in result), sorted(r[0] for r in self.data))
        groups = {}
        for r in result:
            groups.setdefault(tuple(r[i] for i in self.qi_index), []).append(r[self.sa_col])
        for values in groups.values():
            self.assertGreaterEqual(len(values), k)
            self.assertTrue(is_l_diverse(list(Counter(values).values()), l, ldiv_type, c))

    def test_distinct(self):
        for k, l in [(2, 2), (2, 3), (5, 4)]:
            self.check(self.run_tdg(k, l), k, l)

    def test_entropy_recursive(self):
        self.check(self.run_tdg(2, 3, 'entropy'), 2, 3, 'entropy')
        self.check(self.run_tdg(2, 2, 'recursive', 2.0), 2, 2, 'recursive', 2.0)

    def test_sa_not_changed(self):
        result = self.run_tdg(3, 3)
        before = {r[0]: r[self.sa_col] for r in self.data}
        self.assertTrue(all(before[r[0]] == r[self.sa_col] for r in result))

    def test_too_few_values(self):
        with self.assertRaises(ValueError):
            self.run_tdg(2, 50)


class TestReplaceGeneralization(unittest.TestCase):

    def test_numeric_values(self):
        df = pd.DataFrame({'a': ['-122.2', '37.8', '41~48', '12']})
        result = replace_generalization(df, ['a'], qi_index=[0], is_cat=[False])
        self.assertEqual(result['a'].tolist(), [-122.2, 37.8, 44.5, 12.0])


if __name__ == '__main__':
    unittest.main()
