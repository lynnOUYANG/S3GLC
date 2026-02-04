import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch_geometric.utils import to_dense_adj, get_laplacian
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data

def _encode_single_graph(graph, encoding_dim, device=None):
    """
    Helper function to generate spectrum-based encoding for a SINGLE graph.
    This function works on one Data object, not a batch.
    
    Args:
        graph: Data object or dictionary
        encoding_dim: target encoding dimension
        device: target device (if None, uses graph's device)
    """
    # Handle both Data objects and dictionaries
   
        # If graph is a dictionary (from to_data_list)
    
    num_nodes = graph.x['x'].shape[0]
    edge_index = graph.x['edge_index']
    if device is None:
        device = edge_index.device
    
    # Ensure edge_index is on the target device
    edge_index = edge_index.to(device)
    
    if 'edge_weight' in graph.x:
        edge_weight = graph.x['edge_weight'].to(device)
    else:
        edge_weight = None
    if edge_weight is not None:
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
    else:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    # adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    # lap_edge_index, lap_edge_weight = get_laplacian(graph.x['edge_index'], num_nodes=num_nodes)
    #
    #
    # adj = to_dense_adj(edge_index=lap_edge_index,
    #                           edge_attr=lap_edge_weight,
    #                           max_num_nodes=num_nodes)[0]
    eigenvalues = torch.linalg.eigvalsh(adj)
    sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)
    # graph_enc = torch.zeros(encoding_dim, dtype=graph.x['x'].dtype)
    #
    # num_to_copy = min(len(sorted_eigenvalues), encoding_dim)
    # graph_enc[:num_to_copy] = sorted_eigenvalues[:num_to_copy]
    if sorted_eigenvalues.size(0) >= encoding_dim:
        graph_enc = sorted_eigenvalues[:encoding_dim]
    else:
        # Find the position of the smallest positive value
        positive_mask = sorted_eigenvalues > 0
        if positive_mask.any():
            # Find the last (smallest) positive value
            last_positive_idx = torch.where(positive_mask)[0][-1].item()
            # Create the encoding by inserting zeros after the smallest positive value
            graph_enc = torch.zeros(encoding_dim, dtype=sorted_eigenvalues.dtype, device=sorted_eigenvalues.device)
            
            # Copy all eigenvalues up to and including the last positive one
            graph_enc[:last_positive_idx + 1] = sorted_eigenvalues[:last_positive_idx + 1]
            
            # Calculate how many zeros to insert
            zeros_to_insert = encoding_dim - sorted_eigenvalues.size(0)
            
            # Insert zeros after the smallest positive value
            # Shift the remaining eigenvalues (including negatives) to make room for zeros
            remaining_eigenvalues = sorted_eigenvalues[last_positive_idx + 1:]
            if remaining_eigenvalues.size(0) > 0:
                # Place remaining eigenvalues after the inserted zeros
                start_idx = last_positive_idx + 1 + zeros_to_insert
                end_idx = start_idx + remaining_eigenvalues.size(0)
                if end_idx <= encoding_dim:
                    graph_enc[start_idx:end_idx] = remaining_eigenvalues
        else:
            # If no positive values, just pad with zeros at the beginning
            graph_enc = torch.zeros(encoding_dim, dtype=sorted_eigenvalues.dtype, device=sorted_eigenvalues.device)
            graph_enc[:sorted_eigenvalues.size(0)] = sorted_eigenvalues

    return graph_enc


def generate_graph_encoding(graph_list, encoding_dim, device=None, args=None):
    """
    Generates graph-level encodings by iterating through a list of graphs.

    Args:
        graph_list (list[Data]): A list of individual torch_geometric.data.Data objects.
        encoding_dim (int): The target dimension of the graph encoding.
        device: target device (if None, uses first graph's device)
        args: arguments object containing dataset name (args.DS)

    Returns:
        Tensor: The graph encodings, with a shape of [num_graphs, encoding_dim].
    """
    all_encodings = []
    if not graph_list:
        if device is not None:
            return torch.empty(0, encoding_dim, device=device)
        return torch.empty(0, encoding_dim)

    # Save original device (the device that graph_embedding should be on)
    # This should match the device of dataset.data.edge_index
    original_device = device
    
    # Determine device from first graph if not specified
    if device is None and len(graph_list) > 0:
        first_graph = graph_list[0]
        if isinstance(first_graph, dict):
            device = first_graph['x']['edge_index'].device if 'edge_index' in first_graph.get('x', {}) else None
        else:
            device = first_graph.edge_index.device if hasattr(first_graph, 'edge_index') else None
        # If device was None, use the determined device as original_device
        if original_device is None:
            original_device = device

    # For REDDIT-12K dataset, force CPU computation
    computation_device = device
    if args is not None and hasattr(args, 'DS') and args.DS == 'REDDIT-12K' or args.DS == 'ER_MD':
        computation_device = torch.device('cpu')
        print(f"REDDIT-12K dataset detected: forcing CPU computation for graph encoding (will convert back to {original_device})")

    for graph in graph_list:
        try:
            single_graph_encoding = _encode_single_graph(graph, encoding_dim, device=computation_device)
            all_encodings.append(single_graph_encoding)
        except RuntimeError as e:
            # If GPU memory is insufficient, fall back to CPU for this graph
            if "out of memory" in str(e) or "CUDA" in str(e):
                print(f"Warning: GPU OOM for graph, falling back to CPU. Error: {e}")
                single_graph_encoding = _encode_single_graph(graph, encoding_dim, device='cpu')
                single_graph_encoding = single_graph_encoding.to(computation_device) if computation_device is not None else single_graph_encoding
                all_encodings.append(single_graph_encoding)
            else:
                raise

    # Stack all encodings
    graph_embedding = torch.stack(all_encodings, dim=0)
    if args.DS=='PTC_MR' or args.DS=='PTC_MM':
        graph_embedding=graph_embedding/torch.norm(graph_embedding, dim=1, keepdim=True)
    # Convert graph_embedding back to original device if it was computed on a different device
    if original_device is not None and graph_embedding.device != original_device:
        try:
            graph_embedding = graph_embedding.to(original_device)
        except RuntimeError as e:
            # If conversion fails (e.g., GPU unavailable), keep on current device
            print(f"Warning: Cannot move graph_embedding to original device {original_device}, keeping on {graph_embedding.device}. Error: {e}")
    
    return graph_embedding


# class GraphEncoder(nn.Module):
#     def __init__(self, input_dim, output_dim, graph_encoding_dim):
#         super(GraphEncoder, self).__init__()
#         self.graph_encoding_dim = graph_encoding_dim
#
#         # Define layers to process the graph encoding.
#         self.bn = nn.BatchNorm1d(graph_encoding_dim)
#         self.linear = nn.Linear(graph_encoding_dim, graph_encoding_dim)
#
#         # Define a layer to fuse features after concatenation (optional, but recommended).
#         self.fusion_dim = input_dim + graph_encoding_dim
#         self.fusion_layer = nn.Linear(self.fusion_dim, output_dim)
#
#     def forward(self, y, x, edge_index, batch,graph_list):
#
#         graph_enc = generate_graph_encoding(graph_list,self.graph_encoding_dim)
#         print(graph_enc.shape)
#         processed_graph_enc = self.bn(graph_enc)
#         processed_graph_enc = F.relu(self.linear(processed_graph_enc))
#
#         assert y.size(0) == processed_graph_enc.size(0),"Mismatch in number of graphs."
#         combined_y = torch.cat([y, processed_graph_enc], dim=1)
#         fused_y = self.fusion_layer(combined_y)
#
#         return fused_y


def to_data_list(batch, args, device=None):
    """
    Converts a batched Data object back to a list of individual Data objects.
    This is the inverse of the operation performed by the DataLoader.
    
    Args:
        batch: batched Data object
        args: arguments
        device: target device (if None, uses batch's device)
    """
    if not hasattr(batch, 'ptr'):
        # If there's no 'ptr', it's likely a single graph, not a batch.
        return [batch]

    data_list = []
    # The number of graphs in the batch is one less than the length of ptr
    num_graphs = batch.ptr.numel() - 1
    if device is None:
        device = batch.edge_index.device
    ptr = batch.ptr.to(device)
    for i in range(num_graphs):
        node_slice = slice(ptr[i], ptr[i + 1])
       
        edge_mask = (batch.edge_index[0] >= ptr[i]) & (batch.edge_index[0] < ptr[i + 1])
   
        edge_index = batch.edge_index[:, edge_mask]

        remapped_edge_index = edge_index - ptr[i]

        graph_data = {
            'x': batch.x[node_slice],
            'edge_index': remapped_edge_index
        }

        # Extract edge_attr as edge_weight if it exists
        if  args.DS=='MNIST' or args.DS=='CIFAR10':
            edge_attr_masked = batch.edge_attr[edge_mask]
            
            if edge_attr_masked.dim() > 1:
                
                if edge_attr_masked.shape[1] == 1:
                    edge_weight = edge_attr_masked.squeeze(1)
                else:
                    
                    edge_weight = edge_attr_masked[:, 0]
            else:
                
                edge_weight = edge_attr_masked
            graph_data['edge_weight'] = edge_weight
      
        if hasattr(batch, 'y') and batch.y is not None:
            
            if batch.y.size(0) == num_graphs:
                graph_data['y'] = batch.y[i].unsqueeze(0)  # Keep dimension

        data_list.append(Data(graph_data))

    return data_list