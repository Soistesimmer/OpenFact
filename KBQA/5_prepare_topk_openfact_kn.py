import csv
import json
import pickle
import faiss
from tqdm import tqdm


with open('../data/webqsp/webqsp_test.json', 'r') as f:
    raw_data = json.load(f)['Questions']
with open('../data/webqsp/webqsp_train.json', 'r') as f:
    raw_data += json.load(f)['Questions']
questions = {}

def convert_fid_style(key):
    if key[:2] in ['m.', 'g.']:
        key = '/' + key[0] + '/' + key[2:]
    return key

for item in raw_data:
    qid = item['QuestionId']
    answers = set()
    for parse in item['Parses']:
        for answer in parse['Answers']:
            answers.add(convert_fid_style(answer['AnswerArgument']))
    questions[qid] = answers
print(len(questions))

# read kn
relations = {}
oracle = {}
with open('../data/all_entity_pair.tsv', 'r') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for i, line in enumerate(csv_reader):
        id, fid1, fid2, wid1, wid2 = line
        if fid1 in questions[id] or fid2 in questions[id]:
            oracle[id] = 1
        if id not in relations:
            relations[id] = []
        relations[id].append((fid1, fid2, wid1, wid2))
print(len(relations))
print(sum(oracle.values()))

# read kn
kn_texts = {}
with open('../data/all_qa_openfact_kn.tsv', 'r') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for i, line in enumerate(csv_reader):
        if i == 0:
            continue
        id, text, _ = line
        kn_texts[id] = text
print(len(kn_texts))

# read question emb
q_emb = {}
with open('../data/dpr_out_q_emb_0', 'rb') as f:
    num = 0
    for q_id, emb in pickle.load(f):
        q_id = q_id[9:].split('.')[0]
        q_emb[q_id] = emb
        num += 1
print(len(q_emb))

# read relation emb
rel_emb = {}
rel_ids = {}
with open('../data/dpr_all_qa_openfact_emb_0', 'rb') as f:
    num = 0
    for rel_id, emb in pickle.load(f):
        rel_emb[rel_id] = emb
        _i = rel_id.split('.')[0]
        if _i in rel_ids:
            rel_ids[_i].append(rel_id)
        else:
            rel_ids[_i] = [rel_id]
        num += 1
print(len(rel_emb))

entity_mapping = {}
with open('../data/fid2wid.txt', 'r') as f:
    for line in f.readlines():
        fid, wid, ent = line.strip().split('\t')
        entity_mapping[fid] = ent

import numpy as np

def extract_relations_for_question(q_id, candidates, q_emb, rel_emb, rel_ids):
    if q_id not in q_emb:
        return None, None
    _q_emb = q_emb[q_id]
    _rel_ids = []
    _rel_vals = []
    for fid1, fid2, wid1, wid2 in candidates:
        x = rel_ids.get(f'{wid1}||{wid2}', None)
        if x:
            _rel_ids += x
            _rel_vals += [(fid1, fid2)] * len(x)
        else:
            x = rel_ids.get(f'{wid2}||{wid1}', None)
            if x:
                _rel_ids += x
                _rel_vals += [(fid1, fid2)] * len(x)

    if len(_rel_ids) == 0:
        return None, None, None

    if len(_rel_ids) == 1:
        return [0], _rel_vals, _rel_ids

    _rel_embs = np.array([rel_emb[y] for y in _rel_ids])
    index = faiss.IndexFlatIP(768)
    index.add(_rel_embs)
    scores, idxs = index.search(np.expand_dims(_q_emb, 0), min(5000, len(_rel_ids)))
    return scores[0], [_rel_vals[idx] for idx in idxs[0]], [_rel_ids[idx] for idx in idxs[0]]


from tqdm import tqdm
output_data = []
rranks = []
hit_5, hit_25, hit_100 = 0, 0, 0
find_num = 0
cand_num = 0

for q_id, answer in tqdm(questions.items()):
    rrank = 0
    if q_id in relations:
        candidates = relations[q_id]
        scores, idxs, kn_ids = extract_relations_for_question(q_id, candidates, q_emb, rel_emb, rel_ids)
        if scores is not None:
            prev_ent = set()
            real_i = 0
            cand_num += len(idxs)
            for i, idx in enumerate(idxs):
                if set(idx) < prev_ent:
                    continue
                else:
                    real_i += 1
                    prev_ent.update(set(idx))
                if len(answer & set(idx)) > 0:
                    rrank += 1/real_i
                    find_num += 1
                    if real_i <= 5:
                        hit_5 += 1
                    if real_i <= 25:
                        hit_25 += 1
                    if real_i <= 100:
                        hit_100 += 1
                    break
            repeat_text = ''
            for id, score, idx in zip(kn_ids[:200], (scores if isinstance(scores, list) else scores.tolist())[:200], idxs[:200]):
                if kn_texts[id] == repeat_text:
                    continue
                repeat_text = kn_texts[id]
                output_data.append({'qid': q_id, 'openfact': f'[ {entity_mapping[idx[0]]} | {entity_mapping[idx[1]]} ] {kn_texts[id]}', 'score': score})
    rranks.append(rrank)
print(np.mean(rranks), find_num, len(questions), find_num/len(questions), cand_num, cand_num/len(questions))
print(hit_5/len(questions))
print(hit_25/len(questions))
print(hit_100/len(questions))

import jsonlines
with jsonlines.open('../data/selected_openfact_kn.jsonl', 'w') as writer:
    for x in output_data:
        writer.write(x)



