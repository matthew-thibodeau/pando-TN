# Early sandboxing for training and generating TTNs





# Example from docs:
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
>>> Data(edge_index=[2, 4], x=[3, 1])



# Bard:
# import torch
from torch_geometric.nn import GCNConv, MLP

class VectorToGraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.mlp = MLP([output_dim * 2, output_dim])

    def forward(self, x, edge_index):
        # Process input vector (x)
        x = self.mlp(x)

        # Apply GNN layers to extract graph features
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # Generate graph features and edge probabilities
        node_features = self.mlp(x)
        edge_prob = torch.sigmoid(torch.mm(node_features, node_features.T))

        # Return node and edge features for graph construction
        return node_features, edge_prob

# Training loop (replace with your specific loss and optimization)
model = VectorToGraphModel(input_dim=10, hidden_dim=64, output_dim=32)
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(100):
    for data in train_data:
        x, edge_index, y = data.x, data.edge_index, data.y  # Access training data
        node_features, edge_prob = model(x, edge_index)
        # Calculate loss based on node_features, edge_prob, and y (ground truth graph)
        loss = ...
        loss.backward()
        optimizer.step()

# Generate graph from a new input vector
new_x = torch.randn(1, 10)  # Example input vector
node_features, edge_prob = model(new_x, None)  # No edge_index needed for prediction
# Use node_features and edge_prob to construct and analyze the generated graph







