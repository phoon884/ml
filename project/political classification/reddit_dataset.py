import pandas as pd    
from torch.utils.data import Dataset
import torchtext.data as ttd
from torchtext.vocab import GloVe
import torch.nn as nn
import torch

class PoliticalStancesDataset(Dataset):
    def __init__(self, csv_file, tensor_dir, encoded = True):
            if encoded is True:
                self.X = torch.load(tensor_dir)
            if encoded is False:
                df = pd.read_csv(csv_file)
                df['classification'] = df['classification'].map({'left': 0, 'right': 1})
                tonkenized_df = pd.DataFrame()
                tokenizer = ttd.utils.get_tokenizer("toktok")
                for _ , row in df.iterrows(): 
                    js = {
                        "text": tokenizer(row["text"]),
                        "classification": row["classification"]
                    }
                    tonkenized_df = tonkenized_df.append(js,ignore_index = True)
                self.vec  = GloVe("twitter.27B",dim=25)
                length = 50
                self.X = torch.zeros(len(df),length,25)
                for index , row in df.iterrows(): 
                    ret = self.vec.get_vecs_by_tokens(tonkenized_df.iloc[index]["text"], lower_case_backup=True)
                    ret = self.__getvec(ret, 50)
                    self.X[index, :,:] = ret
                torch.save(self.X,tensor_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self):
       return self.X

    def __getvec(tensor, length):
        if tensor.size()[0] > length:
            return tensor[0:length,:]
        m = nn.ZeroPad2d((0, 0, length-tensor.size()[0], 0))
        return m(tensor)