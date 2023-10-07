import json
import jsonlines

import os

WEBQSP_DIR = os.path.dirname(os.path.realpath('./'))
DATA_DIR = os.path.join(WEBQSP_DIR, "data")

# Get entity names from FastRDFStore
# https://github.com/microsoft/FastRDFStore

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

    def unpack(self, fmt, length=1):
        return unpack(fmt, self.readBytes(length))[0]

ALL_ENTITY_NAME_BIN = os.path.join(DATA_DIR, "FastRDFStore", "data", "namesTable.bin")
entity_names = {}
with open(ALL_ENTITY_NAME_BIN, 'rb') as inf:
    stream = BinaryStream(inf)
    dict_cnt = stream.readInt32()
    print("total entities:", dict_cnt)
    for _ in range(dict_cnt):
        key = stream.readString().decode()
        if key.startswith('m.') or key.startswith('g.'):
            key = '/' + key[0] + '/' + key[2:]
        value = stream.readString().decode()
        entity_names[key] = value

entity_mapping = {}
all_num = 0
type_1_num, type_2_num = 0, 0
with jsonlines.open('../data/sparql_search.json', 'r') as reader:
    for line in reader:
        all_num += 1
        fid = line['entity']
        if isinstance(line['response'], str):
            wid = line['response']
            type_1_num += 1
        else:
            if line['response']['results']['bindings']:
                type_2_num += 1
                wid = line['response']['results']['bindings'][0]['item']['value'].split('/')[-1]
            else:
                wid = None
        if wid:
            entity_mapping[fid] = wid

with open('../data/fid2wid.txt', 'w') as f:
    for mid, wid in entity_mapping.items():
        f.write(f'{mid}\t{wid}\t{entity_names[mid]}\n')

