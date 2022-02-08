import pandas as pd
from torch.utils.data import Dataset
import torchtext.data as ttd
from torchtext.vocab import GloVe
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader


class PoliticalStancesDataset(Dataset):
    def __init__(self, csv_file, train, random_state=42):
        df = pd.read_csv(csv_file, index_col=0)
        df['classification'] = df['classification'].map(
            {'left': 0, 'right': 1})
        self.text = df["text"].to_numpy()
        self.classification = df["classification"].to_numpy()
        if train is True:
            self.text, _, self.classification, _ = train_test_split(
                self.text, self.classification, test_size=0.25, random_state=random_state)
        else:
            _,  self.text, _, self.classification = train_test_split(
                self.text, self.classification, test_size=0.25, random_state=random_state)
        self.tokenizer = ttd.utils.get_tokenizer("toktok")
        self.vec = GloVe("twitter.27B", dim=25)
        self.X = torch.zeros(50, 25)

    def __len__(self):
        return len(self.classification)

    def __getitem__(self, idx):
        item_tonkenized = self.tokenizer(self.text[idx])
        X = self.vec.get_vecs_by_tokens(
            item_tonkenized, lower_case_backup=True)
        X = self.__getvec(X, 50)
        return X, torch.Tensor([int(self.classification[idx])])

    def __getvec(self, tensor, length):
        if tensor.size()[0] > length:
            return tensor[0:length, :]
        m = nn.ZeroPad2d((0, 0, length-tensor.size()[0], 0))
        return m(tensor)
    def embed(self,text):
        item_tonkenized = self.tokenizer(text)
        X = self.vec.get_vecs_by_tokens(
            item_tonkenized, lower_case_backup=True)
        X = self.__getvec(X, 50)
        
if __name__ == "__main__":
    test = PoliticalStancesDataset("./dataset_finalized.csv", train=True)
    loader = DataLoader(test, batch_size=64, shuffle=True)
    for _, i in loader:
        print(i)
        break
