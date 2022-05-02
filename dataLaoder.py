import os
import torch
from torch.utils.data import Dataset



class MyDataSet (Dataset):
  
    def __init__(self,x,y) :
        super().__init__()

        self.x = torch.from_numpy(x.copy()/255).float()

      # le -1 dans view c'est pour dire que t'adapte la dimension 

        #self.x = self.x.view(self.x.shape[0],-1)
        self.y = torch.from_numpy(y.copy()).float()

        self.n_samples = self.y.shape[0]

    def __getitem__(self, index) :
        return self.x[index],self.y[index]
  
    def __len__(self):
        return self.n_samples
    