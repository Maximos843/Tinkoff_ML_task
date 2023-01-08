import ast
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import argparse
import re


def compare_programs(program1, program2):
    tree1 = ast.parse(program1)
    tree2 = ast.parse(program2)
    features1 = extract_features(tree1)
    features2 = extract_features(tree2)
    for i in features1.keys():
        if features1[i] is None:
          features1[i] = 0
    for i in features2.keys():
        if features2[i] is None:
          features2[i] = 0
    vectorizer = DictVectorizer()
    features1 = vectorizer.fit_transform(features1).toarray()
    features2 = vectorizer.fit_transform(features2).toarray()
    features = np.array([features1, features2]).reshape(2, len(features1[0]))
    model = KMeans(n_clusters=2)
    model.fit(features)
    similarity = 1 if model.predict(features1)[0] == model.predict(features2)[0] else 0
    levenshtein_sim = 1 - levenshtein_distance(program1, program2) / max(len(program1), len(program2))
    score = (similarity + levenshtein_sim) / 2
    return score


def extract_features(node):
    features = {}
    if isinstance(node, ast.AST):
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, ast.AST):
                child_features = extract_features(value)
                for k, v in child_features.items():
                    features[field + '.' + k] = v
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_features = extract_features(item)
                        for k, v in child_features.items():
                            features[field + '.' + k] = v
            else:
                features[field] = value
    return features


parser = argparse.ArgumentParser()
parser.add_argument("input", help="The input file")
parser.add_argument("scores", help="The output file")
args = parser.parse_args()
input_file = args.input
output_file = args.scores
output = open(output_file, "w")
with open(input_file, "r") as f:
    for i in f.readlines():
        buf = i.split()
        f1 = open(buf[0], "r").read()
        f2 = open(buf[1], "r").read()
        output.write(str(round(compare_programs(f1, f2), 4)))
        output.write("\n")
output.close()