import json
from time import sleep

import requests

import os
import jsonlines
from tqdm import tqdm

webqsp_dir = '../webqsp/full'
train_file = os.path.join(webqsp_dir, 'train.json')
test_file = os.path.join(webqsp_dir, 'test.json')
dev_file = os.path.join(webqsp_dir, 'dev.json')

def read_data(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        return [line for line in reader]

# as webqsp does not provide dev set, this file is separated from train file, we combine them as final train file
train_data = read_data(train_file) + read_data(dev_file)
test_data = read_data(test_file)

print(f'training data: {len(train_data)}, testing data: {len(test_data)}')

# collect entities from data
entities = set()
for item in train_data + test_data:
    entities.update(set([x['kb_id'] for x in item['subgraph']['entities']]))
    # entities.update(set([x['kb_id'] for x in item['answers']])) # do not use, it will leak information
    for x in item['subgraph']['tuples']:
        entities.add(str(x[0]['kb_id']))
        entities.add(str(x[-1]['kb_id']))
print(len(entities))

entities = list([f'/{entity[4]}/{entity[6:-1]}' for entity in entities if entity.startswith('<fb:')])

from struct import *

class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readByte(self):
        return self.base_stream.read(1)

    def readBytes(self, length):
        return self.base_stream.read(length)

    def readChar(self):
        return self.unpack('b')

    def readUChar(self):
        return self.unpack('B')

    def readBool(self):
        return self.unpack('?')

    def readInt16(self):
        return self.unpack('h', 2)

    def readUInt16(self):
        return self.unpack('H', 2)

    def readInt32(self):
        return self.unpack('i', 4)

    def readUInt32(self):
        return self.unpack('I', 4)

    def readInt64(self):
        return self.unpack('q', 8)

    def readUInt64(self):
        return self.unpack('Q', 8)

    def readFloat(self):
        return self.unpack('f', 4)

    def readDouble(self):
        return self.unpack('d', 8)

    def decode_from_7bit(self):
        """
        Decode 7-bit encoded int from str data
        """
        result = 0
        index = 0
        while True:
            byte_value = self.readUChar()
            result |= (byte_value & 0x7f) << (7 * index)
            if byte_value & 0x80 == 0:
                break
            index += 1
        return result

    def readString(self):
        length = self.decode_from_7bit()
        return self.unpack(str(length) + 's', length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def writeChar(self, value):
        self.pack('c', value)

    def writeUChar(self, value):
        self.pack('C', value)

    def writeBool(self, value):
        self.pack('?', value)

    def writeInt16(self, value):
        self.pack('h', value)

    def writeUInt16(self, value):
        self.pack('H', value)

    def writeInt32(self, value):
        self.pack('i', value)

    def writeUInt32(self, value):
        self.pack('I', value)

    def writeInt64(self, value):
        self.pack('q', value)

    def writeUInt64(self, value):
        self.pack('Q', value)

    def writeFloat(self, value):
        self.pack('f', value)

    def writeDouble(self, value):
        self.pack('d', value)

    def writeString(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.writeBytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.readBytes(length))[0]

ALL_ENTITY_NAME_BIN = '../data/FastRDFStore/data/namesTable.bin'
entity_names = {}
with open(ALL_ENTITY_NAME_BIN, 'rb') as inf:
    stream = BinaryStream(inf)
    dict_cnt = stream.readInt32()
    print("total entities:", dict_cnt)
    for _ in range(dict_cnt):
        key = stream.readString().decode()
        value = stream.readString().decode()
        if key.startswith('m.') or key.startswith('g.'):
            key = '/' + key[0] + '/' + key[2:]
        entity_names[key] = value

mapping_path='../data/fb2w.nt'
m2w = {}
with open(mapping_path, 'r') as f:
    for line in f.readlines():
        if line.startswith('<http'):
            mid = line.split()[0].split('/')[-1][:-1]
            wid = line.split()[2].split('/')[-1][:-1]
            if mid.startswith('m.') or mid.startswith('g.'):
                mid = '/' + mid[0] + '/' + mid[2:]
            m2w[mid] = wid

last_search_file = '../data/sparql_search.json'
prev_results = {}
with jsonlines.open(last_search_file, 'r') as reader:
    for line in reader:
        prev_results[line['entity']] = line['response']
print(len(prev_results))

writer = jsonlines.open('../data/sparql_search.json', 'w')

search_num = 0
all_num = 0
for entity in tqdm(entities):
    # filter
    if entity in entity_names:
        all_num += 1
        if entity in prev_results:
            writer.write({'entity': entity, 'response': prev_results[entity]})
        elif entity in m2w:
            writer.write({'entity': entity, 'response': m2w[entity]})
        else:
            search_num += 1
            url = 'https://query.wikidata.org/sparql'
            query = f'''
            SELECT ?item ?itemLabel
            WHERE
            {'{'}
              ?item wdt:P646 "{entity}".
              SERVICE wikibase:label {'{'} bd:serviceParam wikibase:language "en". {'}'}
            {'}'}'''
            fail_num = 0
            while fail_num < 3:
                r = requests.get(url, params={'format': 'json', 'query': query})
                if r.status_code == 200:
                    data = r.json()
                    writer.write({'entity': entity, 'response': data})
                    sleep(1)
                    break
                elif r.status_code == 429:
                    fail_num += 1
                    print(fail_num, f'sleep {int(r.headers["Retry-After"])}s')
                    sleep(int(r.headers["Retry-After"]))
                else:
                    fail_num += 1
                    print(fail_num, f'exception error {r.status_code}')
                    sleep(3)

print(search_num, all_num, len(entities))