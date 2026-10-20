import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class NSCL_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, concept_dim=128, num_heads=8):
        super(NSCL_Model, self).__init__()
        
        # 1. GNN Encoder (对应论文的 Heterogeneous Graph Message Passing)
        self.gnn_encoder = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        
        # 2. Concept Pool Aggregator (对应论文的 Concept Vector Pool)
        self.concept_pool = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, concept_dim),
            nn.ReLU()
        )
        
        # 3. Symbolic Layer (对应论文的可微 Lattice Operations)
        # Meet (Conjunction) MLP
        self.meet_mlp = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim)
        )
        
        # Join (Disjunction) MLP
        self.join_mlp = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim)
        )

    def forward(self, x, edge_index):
        # GNN encoding
        h = self.gnn_encoder(x, edge_index)
        h = F.relu(h)
        
        # Concept aggregation
        v_c = self.concept_pool(h)
        
        return v_c
    
    def lattice_meet(self, v_c1, v_c2):
        # Differentiable meet operation
        concat_v = torch.cat([v_c1, v_c2], dim=-1)
        return self.meet_mlp(concat_v)
        
    def lattice_join(self, v_c1, v_c2):
        # Differentiable join operation
        concat_v = torch.cat([v_c1, v_c2], dim=-1)
        return self.join_mlp(concat_v)