from torch.utils.data import Dataset
import shelve
from sklearn.model_selection import train_test_split
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from numpy import zeros, long


class Data(Dataset):
    def __init__(self, file):
        self.file = file
        self.tokens = {"START", "STOP"}
        self.keys = []
        self.key_len = []
        self.max_len = 0
        self.sms = defaultdict(list)
        self.train = []
        self.test = []
        self.int2token = {}
        self.token2int = {}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.get_vector(idx)

    def train_test_sets(self, test_fraction):
        shuffle(self.keys)
        self.train, self.test = train_test_split(self.keys, test_size=test_fraction, random_state=6, shuffle=True)

    def scan(self):
        with shelve.open(self.file, flag="r") as w:
            for k, i in tqdm(w.items()):
                i = i[1]
                self.tokens.update(set(i))
                self.keys.append(k)
                self.key_len.append(len(i))
                self.max_len = max(self.max_len, len(i))
        self.int2token = {n: x for n, x in enumerate(self.tokens, start=1)}
        self.token2int = {x: n for n, x in enumerate(self.tokens, start=1)}
        self.int2token.update({0: "NULL"})
        self.token2int.update({"NULL": 0})
        self.tokens.add("NULL")
        for k, v in sorted(zip(self.key_len, self.keys), key=lambda x: x[0]):
            self.sms[k].append(v)

    def smi2class(self, smile):
        vector = zeros(self.max_len + 1, dtype=long)
        for n in range(self.max_len + 1):
            if len(smile) > n:
                vector[n] = self.token2int[smile[n]]
            else:
                vector[n] = self.token2int['NULL']
        return vector

    def get_vector(self, idx):
        with shelve.open(self.file, flag="r") as w:
            sin = w[idx][1]
        sin.insert(0, "START")
        sin.append("STOP")
        return self.smi2class(sin[1:-1]), self.smi2class(sin[:-1]), self.smi2class(sin[1:])


class DataGen(Dataset):
    def __init__(self, vectors):
        self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


__all__ = ["Data", "DataGen"]
