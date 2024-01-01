import torch
from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(self, index_path, data_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            self.word_index = f.read().split('/')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.read().split('/')
        self.words = []
        for i in range(len(self.data)):
            self.words.append(self.word2index(self.data[i]))

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        input = self.words[idx][:1]
        label = self.words[idx][1:2]
        return (torch.as_tensor(input), torch.as_tensor(label))

    def word2index(self, word):
        wordcode = []
        for char in word:
            wordcode.append(self.word_index.index(char))
        wordcode.append(0)
        return wordcode
