import torch
import pandas as pd
from math import sqrt
from typing import Union
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch.nn.functional import cross_entropy
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_remaining_self_loops
EPS = 1e-15

def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)

class NodeExplainerModule(Module):
    def __init__(self,
                 model = Module,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_edge_size: float = 0.001,
                 coff_edge_ent: float = 0.001,
                 ):
        super(NodeExplainerModule, self).__init__()
        self.coff_edge_size = coff_edge_size
        self.coff_edge_ent = coff_edge_ent
        self.model = model
        self.epochs = epochs
        self.lr = lr
        
    def __set_masks__(self, x: Tensor, edge_index: Tensor, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.node_feat_masks = None
        self.edge_mask = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        loss = cross_entropy_with_logit(raw_preds, x_label)
        m = self.edge_mask.sigmoid()
        loss = loss + self.coff_edge_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_edge_ent * ent.mean()
        return loss
    
    def gnn_explainer_alg(self,
                          graph: Data,
                          ex_label: Tensor,
                          **kwargs
                          ) -> Tensor:
        self.to(graph.x.device)
        self.classifier = kwargs.get('classifier')
        
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)   
        import warnings
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names") 
        for epoch in range(1, self.epochs + 1):
            h = graph.x
            self.model(x=h, edge_index=graph.edge_index)
            graph_features = torch.from_numpy(self.model.get_node_embeddings())
            graph_plase = pd.DataFrame(graph_features.numpy())
            
            df = pd.concat([graph_plase, graph._NLP], axis=0).reset_index(drop=True)
            
            raw_pred = self.classifier.predict_proba(df.T)
            loss = self.__loss__(torch.tensor(raw_pred).to(self.device), ex_label)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            optimizer.step()
            
        return self.edge_mask.data
     
    def forward(self, graph, target_label=None, **kwargs):
        self.model.eval()
        self_loop_edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.x.size(0))
        
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
        edge_masks = []
        
        for ex_label in ex_labels:
            if target_label is None or ex_label.item() == target_label.item():
                self.__clear_masks__()
                self.__set_masks__(graph.x, self_loop_edge_index)
                edge_mask = self.gnn_explainer_alg(graph, ex_label, classifier=kwargs.get('classifier')).sigmoid()
                edge_masks.append(edge_mask)
            
        self.__clear_masks__()
        return edge_masks, self_loop_edge_index
