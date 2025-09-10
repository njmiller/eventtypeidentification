import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Set2Set, global_mean_pool, global_max_pool, GCNConv, Linear, GraphConv
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T

# from sklearn.neighbors import NearestNeighbors
import numpy as np

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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        # self.conv1 = GCNConv(input_dim, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv1 = GraphConv(input_dim, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

"""
For graph-level classification with k-NN graphs, here's the optimal approach:
Recommended Architecture:
1. Graph Attention Network (GAT) backbone

2-3 GAT layers with multi-head attention (4-8 heads)
Residual connections between layers
Layer normalization and dropout for regularization

2. Global pooling layer (this is crucial for graph-level tasks)

Set2Set pooling: Best for small graphs like yours, learns to attend to important nodes
Global attention pooling: Learns weighted combination of all node features
Hierarchical pooling: DiffPool or similar if you want to learn graph structure

3. Final classifier

MLP with 1-2 hidden layers
Dropout for regularization
Softmax output for classification

Architecture flow:
Node features → GAT layers → Global pooling → MLP classifier → Softmax
Key considerations for your setup:
Small graph size advantage: With 10-100 nodes, you can use more sophisticated pooling methods that might be computationally prohibitive on larger graphs.
k-NN edge construction:

Use k=5-10 to balance local structure capture vs. computational efficiency
Consider making edges bidirectional for better information flow
Edge weights based on distance/energy differences can help

Energy features:

Use energy values as initial node features
Consider energy differences as edge features
Normalize energy values across your dataset

Implementation tips:

Use PyTorch Geometric for easy implementation
Start with Set2Set pooling - it often works well for small graphs
Try different aggregation functions (mean, max, attention) in GAT layers
"""

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, heads=4, dropout=0.3):
        super(GraphClassifier, self).__init__()
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        # Additional GAT layers
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        
        # Global pooling layer - Set2Set works well for small graphs
        self.pooling = Set2Set(hidden_dim * heads, processing_steps=3)
        
        # Alternative pooling options (uncomment to try):
        # self.pooling = lambda x, batch: global_mean_pool(x, batch)
        # self.pooling = lambda x, batch: global_max_pool(x, batch)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads * 2, hidden_dim),  # *2 for Set2Set output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # Apply GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            if i == 0:
                x = F.relu(gat_layer(x, edge_index))
            else:
                # Residual connection
                x_new = F.relu(gat_layer(x, edge_index))
                x = x_new + x  # Add residual connection
        
        # Global pooling to get graph-level representation
        x = self.pooling(x, batch)
        
        # Final classification
        x = self.classifier(x)
        
        return x

def construct_knn_graph(points, k=5, include_self=False):
    """
    Construct k-nearest neighbor graph from point coordinates.
    
    Args:
        points: numpy array of shape (n_points, n_features)
        k: number of nearest neighbors
        include_self: whether to include self-loops
    
    Returns:
        edge_index: torch tensor of shape (2, num_edges)
        edge_attr: torch tensor of edge weights (distances)
    """
    nbrs = NearestNeighbors(n_neighbors=k + (1 if not include_self else 0), 
                           algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    if not include_self:
        # Remove self-loops
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    # Create edge list
    edge_list = []
    edge_weights = []
    
    for i in range(len(points)):
        for j in range(k):
            edge_list.append([i, indices[i, j]])
            edge_weights.append(distances[i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_attr

def create_graph_data(points, energies, label, k=5):
    """
    Create PyTorch Geometric Data object from points and energies.
    
    Args:
        points: numpy array of shape (n_points, spatial_dim)
        energies: numpy array of shape (n_points,)
        label: class label for the graph
        k: number of nearest neighbors
    
    Returns:
        Data object for PyTorch Geometric
    """
    # Construct k-NN graph
    edge_index, edge_attr = construct_knn_graph(points, k=k)
    
    # Make graph undirected by adding reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr])
    
    # Create node features (can include both coordinates and energies)
    node_features = torch.cat([
        torch.tensor(points, dtype=torch.float),
        torch.tensor(energies, dtype=torch.float).unsqueeze(1)
    ], dim=1)
    
    # Create graph data
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long)
    )
    
    return data

# Example usage and training loop
def train_model():
    # Example: Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create synthetic dataset
    graphs = []
    num_graphs = 100
    
    for i in range(num_graphs):
        # Generate random points
        n_points = np.random.randint(10, 50)
        points = np.random.randn(n_points, 2)  # 2D points
        
        # Generate energies (correlated with point positions for demo)
        energies = np.sum(points**2, axis=1) + np.random.normal(0, 0.1, n_points)
        
        # Assign labels based on some criterion (e.g., average energy)
        label = 0 if np.mean(energies) < 2.0 else 1
        
        # Create graph
        graph = create_graph_data(points, energies, label, k=5)
        graphs.append(graph)
    
    # Split data
    train_graphs = graphs[:80]
    test_graphs = graphs[80:]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    # Initialize model
    input_dim = 3  # 2D coordinates + energy
    hidden_dim = 64
    num_classes = 2
    
    model = GraphClassifier(input_dim, hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            total += batch.y.size(0)
            correct += (pred == batch.y).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return model

# Run the example
if __name__ == "__main__":
    model = train_model()