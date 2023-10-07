import os
import json
import csv
import jsonlines

qid2fid = {}
with open('../data/STAGG/webquestions.examples.train.e2e.top10.filter.tsv', 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for line in csvreader:
        if line[0] in qid2fid:
            qid2fid[line[0]].add(line[4])
        else:
            qid2fid[line[0]] = {line[4]}
with open('../data/STAGG/webquestions.examples.test.e2e.top10.filter.tsv', 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for line in csvreader:
        if line[0] in qid2fid:
            qid2fid[line[0]].add(line[4])
        else:
            qid2fid[line[0]] = {line[4]}
print(len(qid2fid))

entity_mapping = {}
with open('../data/fid2wid.txt', 'r') as f:
    for line in f.readlines():
        fid, wid = line.strip().split('\t')
        entity_mapping[fid] = wid
print(len(entity_mapping))

webqsp_dir = '../webqsp/full'

train_file = os.path.join(webqsp_dir, 'train.json')
test_file = os.path.join(webqsp_dir, 'test.json')
dev_file = os.path.join(webqsp_dir, 'dev.json')

def read_data(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        return [line for line in reader]

train_data = read_data(train_file)
test_data = read_data(test_file)

print(f'training data: {len(train_data)}, testing data: {len(test_data)}')

def canonicalize(ent):
    if ent.startswith("<fb:m."):
        return "/m/" + ent[6:-1]
    elif ent.startswith("<fb:g."):
        return "/g/" + ent[6:-1]
    else:
        return ent

# collect triple pair
rank_pair = []
pair_set = set()
for line in train_data + test_data:
    id = line['id']
    head_entities = qid2fid.get(id, None)
    if head_entities:
        related_entities = set()
        for x in line['subgraph']['tuples']:
            related_entities.add(canonicalize(str(x[0]['kb_id'])))
            related_entities.add(canonicalize(str(x[-1]['kb_id'])))
        related_entities -= head_entities
        for h_fid in head_entities:
            h_wid = entity_mapping.get(h_fid, None)
            if h_wid:
                for t_fid in related_entities:
                    t_wid = entity_mapping.get(t_fid, None)
                    if t_wid:
                        rank_pair.append([id, h_fid, t_fid, h_wid, t_wid])
                        pair_set.add(' '.join(sorted([h_wid, t_wid])))
print(len(rank_pair), len(pair_set))

with open('../data/all_entity_pair.tsv', 'w') as f:
    csvwriter = csv.writer(f, delimiter='\t')
    csvwriter.writerows(rank_pair)