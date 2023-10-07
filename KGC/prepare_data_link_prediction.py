# coding=utf-8
import csv
import pickle
import pprint

import jsonlines
import numpy as np
import json
from tqdm import tqdm

type = 'codex-m'

entities = []
with open(f'../kg-bert/data/{type}/entities.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            entities.append(line)

triples = set()
with open(f'../kg-bert/data/{type}/test.tsv', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        h, r, t, l = line.split()
        if l == '1':
            triples.add((h, r, t))
        assert h != t

link_prediction = set()
link_prediction.update(triples)
for triple in triples:
    h, r, t = triple
    for entity in entities:
        if entity != t:
            link_prediction.add((entity, r, t))
        if entity != h:
            link_prediction.add((h, r, entity))

print(len(link_prediction))
with jsonlines.open(f'/data/home/antewang/KGC/kg-bert/harddisk/data/{type}_link_prediction.json', 'w') as writer:
    for line in link_prediction:
        writer.write(line)