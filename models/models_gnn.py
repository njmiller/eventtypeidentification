import torch

from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T

class VoxelGraphDataset(Dataset):
    def __init__(self, data, labels, k=2, extra=False):
        self.data = data # x, y, z, e for each hit in each event
        self.labels = labels # type label number for each event

        self.extra = extra

        labels_unique = torch.unique(torch.tensor(labels))

        self.nlabels = len(labels_unique)

        self.label_map = {}
        for i, val in enumerate(labels_unique):
            self.label_map[val] = i

        # Construct the graph from the locations, energy
        # Edges will be constructed using K nearest neighbors
        # given the locations of the nodes
        transform = T.KNNGraph(k=k)
        self.graph_data = []
        for data_tmp, label_tmp in zip(data, labels):

            # x is node level information (energy)
            # y is graph level information (label)
            data0 = Data(pos=data_tmp[:, :3], x=data_tmp[:, 3], y=label_tmp)
            data0 = transform(data0)
            self.graph_data.append(data0) 

    def __len__(self):
        return len(self.data)
    
    def get(self, index):
        return self.graph_data[index]

def gen_testnet1():
    pass
