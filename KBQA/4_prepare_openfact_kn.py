import csv
import json
import pickle

with open('../data/qid2wikiid.json', 'r') as f:
    qid2wikiid = json.load(f)

def find_kn(ent_id, kn):
    for elem in ['sbj', 'obj', 'bnf', 'loc']:
        if f'{elem}_entities' in kn:
            for entity in kn[f'{elem}_entities']:
                if entity['id'] in ent_id:
                    return elem, entity
    raise Exception('Not Found')

# def highlight(text, ent, name):
#     return text.replace(ent['text'], f'''{ent['text']} [ {name} ]''')

import re
def clean_text(text):
    return re.sub('\s+', ' ', text)

def convert_kn_to_text(kn, elem1, elem2, ent1, ent2, ent_name1, ent_name2):
    strs = []
    if 'sbj_str' in kn:
        sbj_raw_text = kn[f'sbj_str']
        # if elem1 == 'sbj':
        #     sbj_raw_text = highlight(sbj_raw_text, ent1, ent_name1)
        # if elem2 == 'sbj':
        #     sbj_raw_text = highlight(sbj_raw_text, ent2, ent_name2)
        strs.append(sbj_raw_text)
    strs.append(kn['rel'])
    if 'obj_str' in kn:
        obj_raw_text = kn[f'obj_str']
        # if elem1 == 'obj':
        #     obj_raw_text = highlight(obj_raw_text, ent1, ent_name1)
        # if elem2 == 'obj':
        #     obj_raw_text = highlight(obj_raw_text, ent2, ent_name2)
        strs.append(obj_raw_text)
    for elem in ['bnf', 'loc']:
        if f'{elem}_str' in kn:
            raw_text = kn[f'{elem}_str']
            if elem in [elem1, elem2]:
                # if elem1 == elem:
                #     raw_text = highlight(raw_text, ent1, ent_name1)
                # if elem2 == elem:
                #     raw_text = highlight(raw_text, ent2, ent_name2)
                strs.append(raw_text)
    return clean_text(' '.join(strs))


entity_mapping = {}
with open('../data/fid2wid.txt', 'r') as f:
    for line in f.readlines():
        fid, wid, ent = line.strip().split('\t')
        entity_mapping[wid] = ent

csvwriter = csv.writer(open('../data/dssk_kn/all_qa_openfact_kn.tsv', 'w'), delimiter='\t')
csvwriter.writerow(['id', 'text', 'title'])

write_num = 0
with open('../data/all_qa_openfact_kn.pkl', 'rb') as f:
    data = pickle.load(f)
    for k, v in data.items():
        wid1, wid2 = k.split('||')
        ent_n1, ent_n2 = entity_mapping[wid1], entity_mapping[wid2]
        wiki_id1 = qid2wikiid.get(wid1, None)
        if wiki_id1:
            wid1 = [wid1] + wiki_id1
        else:
            wid1 = [wid1]
        wiki_id2 = qid2wikiid.get(wid2, None)
        if wiki_id2:
            wid2 = [wid2] + wiki_id2
        else:
            wid2 = [wid2]
        for i, kn in enumerate(v):
            elem1, ent1 = find_kn(wid1, kn)
            elem2, ent2 = find_kn(wid2, kn)
            text = convert_kn_to_text(kn, elem1, elem2, ent1, ent2, ent_n1, ent_n2)
            if len(text.split()) < 150:
                write_num += 1
                csvwriter.writerow([f'{k}.{i}', text, ''])

print(len(data))
print(write_num)
print(sum([len(x) for x in data.values()]))
#
# related_entities = set()
# for k in data:
#     wid1, wid2 = k.split('||')
#     related_entities.add(entity_mapping[wid1])
#     related_entities.add(entity_mapping[wid2])
#
# with open('../data/webqsp/webqsp_test.json', 'r') as f:
#     data = json.load(f)
#
# recall_num = 0
# all_num = 0
# for item in data['Questions']:
#     qid = item['QuestionId']
#     all_num += 1
#     for parse in item['Parses']:
#         find_answer = False
#         for answer in parse['Answers']:
#             answer_id = answer['AnswerArgument']
#             if answer_id[0] in 'mg':
#                 answer_id = '/'+answer_id[0]+'/'+answer_id[2:]
#                 if answer_id in related_entities:
#                     find_answer = True
#                     break
#         if find_answer:
#             recall_num += 1
#             break
# print(recall_num, all_num, recall_num/all_num)
