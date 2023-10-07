import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm

# read entities
with open('../saved_data/codex-s-pred/codex_entity_ids.json', 'r') as f:
    entity_ids = json.load(f)

# read relations
with open('../saved_data/codex-s-pred/codex_relation_ids.json', 'r') as f:
    relation_ids = json.load(f)

# read predictions
# ComplEx results from CodEx official repo
with open('../saved_data/codex-s-pred/codex_test_predictions.json', 'r') as f: 
    predictions = json.load(f)

# read reranker prediction
reranker_predictions = {}
with open('../saved_data/codex-m_results/lp_test_predict_results.txt', 'r') as f:
    for line in f.readlines():
        _, head, rel, tail, score = line.strip().split()
        # head = entity_ids.index(head)
        # rel = relation_ids.index(rel)
        # tail = entity_ids.index(tail)
        reranker_predictions[(head, rel, tail)] = float(score)

# read raw triples
data_dir = '../saved_data/codex/data'
triple_dir = os.path.join(data_dir, 'triples/codex-s')
triple_valid_file = os.path.join(triple_dir, 'test.txt')
with open(triple_valid_file, "r") as f:
    triples = []
    for line in f.readlines():
        _triple = list(line.strip().split())
        assert len(_triple) == 3
        triples.append(tuple(_triple))

# read raw triples
triple_valid_file = os.path.join(triple_dir, 'train.txt')
eval_triples = [] + triples
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
_triples = [(entity_ids[pred['triple'][0]], relation_ids[pred['triple'][1]], entity_ids[pred['triple'][2]]) for pred in
            predictions]
assert len(set(triples) - set(_triples)) == 0

sp_rel_dict = {}
po_rel_dict = {}
for x in eval_triples:
    x = list(x)
    x[0] = entity_ids.index(x[0])
    x[1] = relation_ids.index(x[1])
    x[2] = entity_ids.index(x[2])
    if (x[0], x[1]) in sp_rel_dict:
        sp_rel_dict[(x[0], x[1])].append(x[2])
    else:
        sp_rel_dict[(x[0], x[1])] = [x[2]]
    if (x[1], x[2]) in po_rel_dict:
        po_rel_dict[(x[1], x[2])].append(x[0])
    else:
        po_rel_dict[(x[1], x[2])] = [x[0]]

sp_rerank_dict = {}
po_rerank_dict = {}
for x, s in reranker_predictions.items():
    x = list(x)
    x[0] = entity_ids.index(x[0])
    x[1] = relation_ids.index(x[1])
    x[2] = entity_ids.index(x[2])
    if (x[0], x[1]) in sp_rerank_dict:
        sp_rerank_dict[(x[0], x[1])].append((x[2], s))
    else:
        sp_rerank_dict[(x[0], x[1])] = [(x[2], s)]
    if (x[1], x[2]) in po_rerank_dict:
        po_rerank_dict[(x[1], x[2])].append((x[0], s))
    else:
        po_rerank_dict[(x[1], x[2])] = [(x[0], s)]

# check ranking
rank_all = []
hit_1, hit_10 = 0, 0
for pred in predictions:
    triple = pred['triple']
    s, p, o = triple[0], triple[1], triple[-1]
    scores = torch.tensor(pred['scores'])
    # scores = torch.zeros_like(scores) # do not consider ComplEx score
    sp_scores = scores[:len(pred['scores']) // 2]
    po_scores = scores[len(pred['scores']) // 2:]
    if (s, p) in sp_rerank_dict:
        for x in sp_rerank_dict[(s, p)]:
            sp_scores[x[0]] += x[1]
    if (p, o) in po_rerank_dict:
        for x in po_rerank_dict[(p, o)]:
            po_scores[x[0]] += x[1]
    if (s, p) in sp_rel_dict:
        for x in sp_rel_dict[(s, p)]:
            if x != o:
                sp_scores[x] = -1e6
    if (p, o) in po_rel_dict:
        for x in po_rel_dict[(p, o)]:
            if x != s:
                po_scores[x] = -1e6
    rank_o = torch.sum(sp_scores > sp_scores[o])
    rank_s = torch.sum(po_scores > po_scores[s])
    rank_all.append(rank_o)
    rank_all.append(rank_s)
    if rank_o < 1:
        hit_1 += 1
    if rank_s < 1:
        hit_1 += 1
    if rank_o < 10:
        hit_10 += 1
    if rank_s < 10:
        hit_10 += 1

rank_all = np.array(rank_all)
# print(np.sum(rank_all < 200))
print(np.mean(1 / (rank_all + 1)))
print(hit_1/len(predictions)/2, hit_10/len(predictions)/2)
# print(np.mean(rank_all) + 1)
