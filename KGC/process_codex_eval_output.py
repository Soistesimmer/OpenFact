'''
This script is to use to get the top 200 candidates from ComplEx outputs
'''
import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
import sys

size = sys.argv[1]
# read entities
with open(f'../saved_data/codex-{size}-pred/codex_entity_ids.json', 'r') as f:
    entity_ids = json.load(f)

# read relations
with open(f'../saved_data/codex-{size}-pred/codex_relation_ids.json', 'r') as f:
    relation_ids = json.load(f)

# read predictions
with open(f'../saved_data/codex-{size}-pred/codex_test_predictions.json', 'r') as f:
    predictions = json.load(f)

# read raw triples
data_dir = '../saved_data/codex/data'
triple_dir = os.path.join(data_dir, f'triples/codex-{size}')
triple_valid_file = os.path.join(triple_dir, 'test.txt')
with open(triple_valid_file, "r") as f:
    triples = []
    for line in f.readlines():
        _triple = list(line.strip().split())
        assert len(_triple) == 3
        triples.append(tuple(_triple))

# read raw triples
triple_valid_file = os.path.join(triple_dir, 'train.txt')
eval_triples = []
with open(triple_valid_file, "r") as f:
    for line in f.readlines():
        _triple = list(line.strip().split())
        assert len(_triple) == 3
        eval_triples.append(tuple(_triple))
triple_valid_file = os.path.join(triple_dir, 'valid.txt')
with open(triple_valid_file, "r") as f:
    for line in f.readlines():
        _triple = list(line.strip().split())
        assert len(_triple) == 3
        eval_triples.append(tuple(_triple))

# check consist
_triples = [(entity_ids[pred['triple'][0]], relation_ids[pred['triple'][1]], entity_ids[pred['triple'][2]]) for pred in predictions]
assert len(set(triples) - set(_triples)) == 0

# prepare data
topk = 200 # large number to cover positive ones
output_triples = set()
for pred in tqdm(predictions):
    triple = pred['triple']
    s, p, o = triple[0], triple[1], triple[-1]
    scores = torch.tensor(pred['scores'])
    sp_scores = scores[:len(pred['scores'])//2]
    po_scores = scores[len(pred['scores'])//2:]
    sp_index = torch.argsort(sp_scores, descending=True)[:topk]
    po_index = torch.argsort(po_scores, descending=True)[:topk]

    for x in sp_index:
        output_triples.add((entity_ids[s], relation_ids[p], entity_ids[x]))
    for x in po_index:
        output_triples.add((entity_ids[x], relation_ids[p], entity_ids[o]))
print(len(output_triples))
output_triples = list(output_triples - set(eval_triples))
print(len(output_triples))

with open(f'../saved_data/codex-{size}-pred/test.tsv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter="\t")
    csvwriter.writerows([list(x)+[0] for x in output_triples])

# # check ranking
# rank_all = []
# for pred in predictions:
#     triple = pred['triple']
#     s, o = triple[0], triple[-1]
#     scores = torch.tensor(pred['scores'])
#     sp_scores = scores[:len(pred['scores'])//2]
#     po_scores = scores[len(pred['scores'])//2:]
#     rank_o = torch.sum(sp_scores > sp_scores[o])
#     rank_s = torch.sum(po_scores > po_scores[s])
#     rank_all.append(rank_o)
#     rank_all.append(rank_s)
#
# print(np.mean(rank_all) + 1)