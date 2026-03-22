import pandas as pd
import joblib
import os
from multiprocessing import Pool
import re
from difflib import SequenceMatcher  # for longest common substring
from functools import partial
from operator import itemgetter
import Levenshtein  # levenstein/edit distance; docs here: https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html
import numpy as np
from sklearn.metrics import roc_auc_score

data = pd.read_csv('../../Potomac/AIT 620/LabWorks/PythonProject/datafiles/forbes_freebase_goldstandard_train.csv', names=['string1', 'string2', 'matched'])

data.head()

len(data)

def clean_string(string):
    '''We will use this functions to remove special characters etc before
    any string distance calculation.
    '''
    return ''.join(map(lambda x: x.lower() if str.isalnum(x) else ' ', string)).strip()

def levenstein_distance(s1_, s2_):
    s1, s2 = clean_string(s1_), clean_string(s2_)
    len_s1, len_s2 = len(s1), len(s2)
    return Levenshtein.distance(
        s1, s2
    ) / max([len_s1, len_s2])

def jaro_winkler_distance(s1_, s2_):
    s1, s2 = clean_string(s1_), clean_string(s2_)
    return Levenshtein.jaro_winkler(s1, s2)

def common_substring_distance(s1_, s2_):
    s1, s2 = clean_string(s1_), clean_string(s2_)
    len_s1, len_s2 = len(s1), len(s2)
    match = SequenceMatcher(
        None, s1, s2
    ).find_longest_match(0, len_s1, 0, len_s2)
    len_s1, len_s2 = len(s1), len(s2)
    norm = max([len_s1, len_s2])
    return 1 - min([1, match.size / norm])


dists = np.zeros(shape=(len(data), 3))
for algo_i, algo in enumerate(
        [levenstein_distance, jaro_winkler_distance, common_substring_distance]
):
    for i, string_pair in data.iterrows():
        dists[i, algo_i] = algo(string_pair['string1'], string_pair['string2'])

    print('AUC for {}: {}'.format(
        algo.__name__,
        roc_auc_score(data['matched'].astype(float), 1 - dists[:, algo_i])
    ))