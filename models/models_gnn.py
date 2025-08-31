import torch

from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T

class GraphDataset(Dataset):
    def __init__(self, data, labels, k=5, transform=None):
        super().__init__(transform=transform)
        print("Loading with KNNGraph")
        self.data = data # x, y, z, e for each hit in each event
        self.labels = labels # type label number for each event

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
        i = 0
        for data_tmp, label_tmp in zip(data, labels):
            data_tmp = torch.tensor(data_tmp, dtype=torch.float)
            label_tmp = torch.tensor(label_tmp, dtype=torch.float)

            i += 1
            if i % 40000 == 0:
                print("i:", i)

            # x is node level information (energy)
            # y is graph level information (label)
            # data0 = Data(pos=data_tmp[:, :3], x=data_tmp[:, 3], y=label_tmp)
            data0 = Data(pos=data_tmp[:, :3], x=data_tmp, y=label_tmp)
            # data0 = transform(data0)
            self.graph_data.append(data0) 

    def __len__(self):
        return len(self.data)
    
    def len(self):
        return len(self.data)
    
    def get(self, index):
        return self.graph_data[index]
