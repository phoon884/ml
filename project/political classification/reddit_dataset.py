import pandas as pd    
from torch.utils.data import Dataset
class PoliticalStancesDataset(Dataset):
    def __init__(self, csv_file, tensor_dir, encoded = True):
            if encoded is True:
                pass
            if encoded is False:
                df = pd.read_csv(csv_file)
                df['classification'] = df['classification'].map({'left': 0, 'right': 1})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
       pass