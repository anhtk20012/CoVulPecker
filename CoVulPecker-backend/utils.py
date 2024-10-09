import re
import os
import gc
import torch
import joblib
import shutil
import numpy as np
import pandas as pd
import torch_scatter
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from node2vec import Node2Vec
import torch.nn.functional as F
from run_docker import Joern2Dot
from torch_geometric.nn import GATv2Conv
from Explainer import NodeExplainerModule
from torch.nn import Module, ReLU, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from networkx.drawing.nx_pydot import read_dot
from transformers import AutoTokenizer, AutoModel, logging
from torch_geometric.utils import remove_self_loops, coalesce, add_remaining_self_loops, to_dense_adj

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
# warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")
        
class graph_matrix():
    def extract_sub_label(self, label):
        match = re.search(r'<SUB>(\d+)</SUB>', label)
        if match:
            return match.group(1)
        return None
    
    def relabel_data(self, graph):
        for node, data in graph.nodes(data=True):
            old_label = data.get('label')
            if old_label:
                line_number = self.extract_sub_label(old_label)
                if line_number:
                    graph.nodes[node]['label'] = line_number
                
    def make_graph(self, dot_file):
        G = read_dot(dot_file)
        mapping = { old_name: idx for idx, old_name in enumerate(G.nodes()) }
        new_G = nx.relabel_nodes(G, mapping)
        self.relabel_data(new_G)
        return new_G

    def make_matrix(self, graph):
        all_edge_index = []
        source_nodes = []
        target_nodes = []

        for source, target in graph.edges(): 
            source_nodes.append(source)
            target_nodes.append(target)

        all_edge_index.append(source_nodes)
        all_edge_index.append(target_nodes)

        return np.array(all_edge_index)

class Preprocess():    
    def remove_comments(self, string):
        pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
        regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
        def _replacer(match):
            if match.group(2) is not None:
                return ""
            else:
                return match.group(1)
        return regex.sub(_replacer, string)

    def rename_funcs(self, source_code):
        matches = re.findall(r'(\w+)\(', source_code)
        functions = []
        for m in matches:
            if m not in functions and bool(re.match(r'^[a-z]+$', m)) == False:
                functions.append(m)
        idx = 0
        for f in functions:
            if idx == 0:
                source_code = source_code.replace(f, 'function')
            else:
                source_code = source_code.replace(f, f'func_{idx}')
            idx = idx + 1
        return source_code

    def rename_vars(self, source_code):
        matches = re.findall(r'\b(\w+)\b\s*=', source_code)
        variables = []
        for m in matches:
            if m not in variables and len(m) > 1:
                variables.append(m)
        idx = 1
        for f in variables:
            source_code = source_code.replace(f, f'var_{idx}')
            idx = idx + 1
        return source_code

    def clean_code(self, source_code):
        source_code = self.remove_comments(source_code)
        source_code = self.rename_funcs(source_code)
        source_code = self.rename_vars(source_code)
        return source_code
        
class GraphEmbedding():  
    def __init__(self, graph, num_dimension, walk_length=100, num_walks=10, workers=1):
        self.w2v_model = Node2Vec(graph, dimensions=num_dimension,
                                  walk_length=walk_length, num_walks=num_walks,
                                  p=1, q=1, workers=workers, seed=42).fit(window=10, min_count=1)
            
        self.embedding_graphs = {}  

    def get_embeddings(self, graph: nx.classes.graph.Graph):
        self.embedding_graphs = {word: self.w2v_model.wv[word] for word in graph.nodes()}
        
        results = np.array(list(self.embedding_graphs.values()))    

        return results 

class GraphCodeBERT():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.embedding_graphs = {}
        
    def get_embeddings(self, graph: nx.DiGraph):
        for _, data in graph.nodes(data=True):
            word = data['label']
            
            tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(word) + [self.tokenizer.sep_token]
            tokens_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))[None, :].to(self.device)
            
            with torch.no_grad():
                feature = self.model(tokens_tensor)[0][0, 0].cpu()
            self.embedding_graphs[word] = feature.detach().numpy()
            
        results = np.array(list(self.embedding_graphs.values()))
        
        mean_tensor = [torch.mean(torch.from_numpy(embedding)).item() for embedding in results]
        mean_tensor = mean_tensor[:200] + [0] * (200 - len(mean_tensor))
        
        return mean_tensor

class GAT(Module):
    def __init__(self, embed: torch.Tensor, in_features: int, n_hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.embeddings = embed
        
        self.layer1 = GATv2Conv(in_features, n_hidden, heads=1)
        self.layer2 = GATv2Conv(n_hidden, n_hidden, heads=n_heads)
        self.layer3 = GATv2Conv(n_hidden * n_heads, in_features, heads=1)
        
        self.activation = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.layer1(x, edge_index)
        x = self.activation(x)
        
        x = self.layer2(x, edge_index)
        x = self.activation(x)
        
        x = self.layer3(x, edge_index)
        
        x = self.dropout(x)
        self.embeddings = x
        
        return x

    def get_node_embeddings(self):
        mean_tensor = [torch.mean(embedding).item() for embedding in self.embeddings]
        mean_tensor = mean_tensor[:200] + [0] * (200 - len(mean_tensor))

        return np.array(mean_tensor)

class Gnnexplainer_run():
    def __init__(self, device, graph, model, classifier):
        self.device = device
        self.graph = graph
        self.model = model
        self.classifier = classifier
        
    def explainer_model(self, exp):
        explainer = NodeExplainerModule(model=self.model, epochs=800, lr=0.05)
        explainer.device = self.device
        
        edge_index, _ = remove_self_loops(self.graph.edge_index)
        edge_index = coalesce(edge_index)
        
        if edge_index.shape[1] != 0:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.graph.x.shape[0])
            edge_masks, self_loop_edge_index = explainer(
                self.graph, None, num_classes=2, classifier=self.classifier
                )
            edge_weight = edge_masks[torch.argmax(exp, dim=-1)]
            edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())   
            self.graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
            
        return self.graph
        
class VulnDetectorExplainer():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_file = Path(__file__).parent
        self.input = os.path.join(self.path_file, "Input")
        os.makedirs(self.input, exist_ok=True)
        self.output = os.path.join(self.path_file, "Output")
        os.makedirs(self.output, exist_ok=True)
        
        logging.set_verbosity_error()
        self.GCBtokenizer = AutoTokenizer.from_pretrained(os.path.join(self.path_file, "Model", "FTGraphCodeBert.pt"))
        self.GCBmodel = AutoModel.from_pretrained(os.path.join(self.path_file, "Model", "FTGraphCodeBert.pt"))
        self.GCBmodel.to(self.device)
        
        self.classifier = joblib.load(os.path.join(self.path_file, "Model", "randomforest.joblib"))
    
    def source2dot(self, source):
        tmp = os.path.join(self.path_file, 'tmp')
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        
        cleaned_source = Preprocess().clean_code(source)
        with open(os.path.join(tmp, "code.c"), "w") as file:
            file.write(cleaned_source.replace('\r\n', '\n'))
        
        Joern2Dot(__file__)
        
        dot_file = os.path.join(tmp, "code.dot")
        dot_files = os.listdir(dot_file)
        for dot in dot_files:
            full_path = os.path.join(dot_file, dot)
            with open(full_path, 'r') as file:
                first_line = file.readline().strip()

            if 'digraph "function" {' in first_line:
                dest_path = os.path.join(self.output, "graph.dot")
                shutil.move(full_path, dest_path)
                shutil.rmtree(tmp)
                os.mkdir(tmp)
                break
            
        return dest_path
            
    def graph_phase(self, graph, matrix):
        with torch.no_grad():
            graphembedd_model = GraphEmbedding(graph, num_dimension=200)
            embeddings = graphembedd_model.get_embeddings(graph)
            
        embeddings = torch.tensor(embeddings).to(self.device)
        adj_tensor = torch.tensor(matrix).to(self.device)
            
        experiment_model = GAT(embed=embeddings, in_features=200, n_hidden=128,
                                                  n_heads=8, dropout=0.1).to(self.device)

        optimizer = torch.optim.Adam(experiment_model.parameters(), lr=0.01, weight_decay = 5e-4)
        
        epochs = 10
        for e in range(epochs):
            experiment_model.train()
            optimizer.zero_grad()
            outputs = experiment_model(embeddings, adj_tensor)
            embeddings = outputs
        
        graph_features = experiment_model.get_node_embeddings()
        graph_features = torch.from_numpy(graph_features)
        
        df = pd.DataFrame(graph_features.numpy())
        list_data = [df, embeddings, adj_tensor]
        
        del experiment_model, graphembedd_model, embeddings, adj_tensor, df
        torch.cuda.empty_cache()
        gc.collect()
        
        return list_data
    
    def nlp_phase(self, graph):
        with torch.no_grad():
            model = GraphCodeBERT(self.GCBmodel, self.GCBtokenizer, self.device)
            embeddings = model.get_embeddings(graph)
            GCB_embed = pd.DataFrame(embeddings)
        
        del model, embeddings
        torch.cuda.empty_cache()
        gc.collect()
        
        return GCB_embed
    
    def run_detector(self, source: str):
        dot_file = self.source2dot(source)
        # dot_file = os.path.join(self.output, 'graph.dot')
        graph = graph_matrix().make_graph(dot_file)
        matrix = graph_matrix().make_matrix(graph)
        graph_plase_train = self.graph_phase(graph, matrix)
        nlp_phase_train = self.nlp_phase(graph)
        df = pd.concat([graph_plase_train[0], nlp_phase_train], axis=0).reset_index(drop=True)
        pred = self.classifier.predict(df.T)
        print(graph_plase_train[2])
        if pred == 1:
            prob = self.classifier.predict_proba(df.T)
            exp_prob_label = F.one_hot(torch.argmax(torch.tensor(prob).to(self.device), dim=-1), 2)
            graph_data = Data(x=graph_plase_train[1], edge_index=graph_plase_train[2])
            graph_data.__setitem__("_NLP", nlp_phase_train)
            graph_data.__setitem__("_NODE", [graph.nodes[node]['label'] for node in graph.nodes])
            model = GAT(embed=graph_plase_train[1], in_features=200, n_hidden=128, n_heads=8, dropout=0.1).to(self.device)
            
            explainer = Gnnexplainer_run(self.device, graph_data, model, self.classifier)
            data_final = explainer.explainer_model(exp_prob_label)
            data_final.to(self.device)
            edge_weight = data_final.edge_weight
            if len(edge_weight) > 10:
                value, index = torch.topk(edge_weight, k=10)
            else:
                index = torch.arange(edge_weight.shape[0])
            
            temp = torch.zeros_like(edge_weight).to(data_final.edge_index.device)
            temp[index] = edge_weight[index]
            adj_mask = torch.sparse_coo_tensor(data_final.edge_index, temp, [data_final.x.shape[0], data_final.x.shape[0]])
            adj_mask_binary = to_dense_adj(data_final.edge_index[:, temp != 0].long(), max_num_nodes=data_final.x.shape[0]).squeeze(0)
            out_degree = torch.sum(adj_mask_binary, dim=1)
            out_degree[out_degree == 0] = 1e-8
            in_degree = torch.sum(adj_mask_binary, dim=0)
            in_degree[in_degree == 0] = 1e-8
            
            line_importance_init = torch.ones(data_final.x.shape[0]).unsqueeze(-1).to(self.device)
            line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
            line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
            line_importance = line_importance_out + line_importance_in
            lines = data_final._NODE
            ret = sorted(
                list(
                    zip(
                        line_importance.squeeze(-1).cpu().numpy(),
                        lines,
                    )
                ),
                reverse=True,
            )
            filtered_ret = []
            for i in ret:
                if i[0] > 0:
                    filtered_ret.append(int(i[1]))
            unique_list = list(set(filtered_ret))
            return ["True",unique_list]
        else:
            return ["False",[]]
        