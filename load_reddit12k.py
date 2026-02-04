"""
Load REDDIT-12K dataset and convert it to TUDataset format (without aug_data).
This script reads the raw REDDIT-12K files and creates a PyTorch Geometric InMemoryDataset.
The processed data will be saved to processed/data.pt for faster loading next time.

File format:
- REDDIT-MULTI-12K.edges: Each line is "node1,node2" (1-based indexing)
- REDDIT-MULTI-12K.graph_idx: Each line is a graph ID indicating which graph the node belongs to (1-based indexing)
- REDDIT-MULTI-12K.graph_labels: Each line is a graph label (1-based indexing)
"""

import os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data


class Reddit12KDataset(InMemoryDataset):
    """
    REDDIT-12K dataset loader compatible with TUDataset format.
    Processes raw files and saves to processed/data.pt for faster loading.
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root: Root directory containing REDDIT-12K raw files or processed data
            transform: Optional transform to apply to data
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
        """
        self.raw_dir_name = 'REDDIT-12K'
        super(Reddit12KDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        # Check if processed data exists
        if not osp.exists(self.processed_paths[0]):
            # Process data if not exists
            print(f"Processed data not found at {self.processed_paths[0]}")
            print("Processing raw data...")
            self._process()
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        """List of raw file names that should be found in raw_dir"""
        return [
            'REDDIT-MULTI-12K.edges',
            'REDDIT-MULTI-12K.graph_idx',
            'REDDIT-MULTI-12K.graph_labels'
        ]
    
    @property
    def processed_file_names(self):
        """List of processed file names"""
        return ['data.pt']
    
    @property
    def raw_dir(self):
        """Directory containing raw files"""
        return osp.join(self.root, self.raw_dir_name)
    
    @property
    def processed_dir(self):
        """Directory containing processed files"""
        return osp.join(self.root, self.raw_dir_name, 'processed')
    
    def _process(self):
        """Process raw files and save to processed/data.pt (same as TUDataset_aug.process)"""
        print("Processing REDDIT-12K dataset...")
        
        # Step 1: Read graph labels (one label per graph)
        graph_labels_file = osp.join(self.raw_dir, 'REDDIT-MULTI-12K.graph_labels')
        print("  Reading graph labels...")
        graph_labels = []
        with open(graph_labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    graph_labels.append(int(line))
        graph_labels = np.array(graph_labels, dtype=np.int64)
        num_graphs = len(graph_labels)
        print(f"    Found {num_graphs} graphs")
        
        # Step 2: Read graph_idx (which graph each node belongs to)
        graph_idx_file = osp.join(self.raw_dir, 'REDDIT-MULTI-12K.graph_idx')
        print("  Reading graph indices...")
        graph_idx = []
        with open(graph_idx_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    graph_idx.append(int(line))
        graph_idx = np.array(graph_idx, dtype=np.int64)
        num_nodes_total = len(graph_idx)
        print(f"    Found {num_nodes_total} nodes")
        
        # Step 3: Read edges
        edges_file = osp.join(self.raw_dir, 'REDDIT-MULTI-12K.edges')
        print("  Reading edges...")
        edges = []
        with open(edges_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        node1, node2 = int(parts[0]), int(parts[1])
                        edges.append([node1, node2])
        edges = np.array(edges, dtype=np.int64)
        print(f"    Found {len(edges)} edges")
        
        # Step 4: Convert from 1-based to 0-based indexing
        print("  Converting to 0-based indexing...")
        graph_idx = graph_idx - 1  # Convert graph IDs to 0-based
        graph_labels = graph_labels - 1  # Convert labels to 0-based
        edges = edges - 1  # Convert node IDs to 0-based
        
        # Validate indices
        assert graph_idx.min() >= 0, f"Graph index minimum is {graph_idx.min()}, expected >= 0"
        assert edges.min() >= 0, f"Edge index minimum is {edges.min()}, expected >= 0"
        assert graph_idx.max() < num_graphs, f"Graph index maximum is {graph_idx.max()}, expected < {num_graphs}"
        
        # Step 5: Group nodes by graph_id
        print("  Grouping nodes by graph...")
        graph_nodes = {}
        for global_node_idx, graph_id in enumerate(graph_idx):
            if graph_id not in graph_nodes:
                graph_nodes[graph_id] = []
            graph_nodes[graph_id].append(global_node_idx)
        
        # Step 6: Process each graph and create data_list
        print("  Processing graphs...")
        data_list = []
        edges_array = edges
        
        for graph_id in range(num_graphs):
            if graph_id not in graph_nodes:
                # Empty graph, skip
                continue
            
            # Get nodes belonging to this graph
            global_node_indices = np.array(graph_nodes[graph_id], dtype=np.int64)
            num_nodes = len(global_node_indices)
            
            # Create mapping from global node index to local node index (0-based)
            max_global_node_idx = global_node_indices.max()
            global_to_local = np.full(max_global_node_idx + 1, -1, dtype=np.int64)
            global_to_local[global_node_indices] = np.arange(num_nodes, dtype=np.int64)
            
            # Extract edges for this graph
            # An edge belongs to this graph if both endpoints are in global_node_indices
            src_mask = np.isin(edges_array[:, 0], global_node_indices)
            dst_mask = np.isin(edges_array[:, 1], global_node_indices)
            edge_mask = src_mask & dst_mask
            
            if np.any(edge_mask):
                # Get edges belonging to this graph
                graph_edges_global = edges_array[edge_mask]
                # Map global node indices to local indices
                local_src = global_to_local[graph_edges_global[:, 0]]
                local_dst = global_to_local[graph_edges_global[:, 1]]
                edge_list = np.stack([local_src, local_dst], axis=1)
            else:
                # Graph with no edges, create self-loops for all nodes
                edge_list = np.array([[i, i] for i in range(num_nodes)], dtype=np.int64)
            
            # Convert to torch tensor (shape: [2, num_edges])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Create node features (default: ones, as REDDIT-12K doesn't have node features)
            x = torch.ones((num_nodes, 1), dtype=torch.float32)
            
            # Get graph label (scalar tensor)
            y = torch.tensor(graph_labels[graph_id], dtype=torch.long)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            
            if (graph_id + 1) % 1000 == 0:
                print(f"    Processed {graph_id + 1}/{num_graphs} graphs...")
        
        print(f"  Total graphs: {len(data_list)}")
        
        # Step 7: Collate data_list into data and slices (same as TUDataset)
        print("  Collating data...")
        self.data, self.slices = self.collate(data_list)
        
        # Step 8: Apply pre_filter if provided (same as TUDataset_aug.process)
        if self.pre_filter is not None:
            print("  Applying pre_filter...")
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)
        
        # Step 9: Apply pre_transform if provided (same as TUDataset_aug.process)
        if self.pre_transform is not None:
            print("  Applying pre_transform...")
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        
        # Step 10: Save processed data (same as TUDataset_aug.process)
        print(f"  Saving to {self.processed_paths[0]}...")
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print("  Processing complete!")


def load_reddit12k_dataset(data_path):
    """
    Load REDDIT-12K dataset from raw files or processed data.
    
    Args:
        data_path: Path to the directory containing REDDIT-12K raw files or processed data
    
    Returns:
        dataset: InMemoryDataset object compatible with TUDataset format
    """
    dataset = Reddit12KDataset(root=data_path)
    return dataset


if __name__ == '__main__':
    # Test loading
    data_path = '/home/comp/csxylin/data'
    dataset = load_reddit12k_dataset(data_path)
    
    print(f"\nDataset loaded successfully!")
    print(f"  - Number of graphs: {len(dataset)}")
    print(f"  - Processed file: {dataset.processed_paths[0]}")
    print(f"  - Data keys: {list(dataset.data.keys) if hasattr(dataset.data, 'keys') else 'N/A'}")
    print(f"  - Slices keys: {list(dataset.slices.keys()) if hasattr(dataset.slices, 'keys') else 'N/A'}")
    print(f"\nFirst graph info:")
    first_data = dataset[0]
    print(f"  - Has x: {hasattr(first_data, 'x') and first_data.x is not None}")
    print(f"  - x shape: {first_data.x.shape if hasattr(first_data, 'x') and first_data.x is not None else 'N/A'}")
    print(f"  - Has edge_index: {hasattr(first_data, 'edge_index') and first_data.edge_index is not None}")
    print(f"  - edge_index shape: {first_data.edge_index.shape if hasattr(first_data, 'edge_index') and first_data.edge_index is not None else 'N/A'}")
    print(f"  - Has y: {hasattr(first_data, 'y') and first_data.y is not None}")
    print(f"  - y value: {first_data.y.item() if hasattr(first_data, 'y') and first_data.y is not None else 'N/A'}")
    print(f"\nDataset.data.y shape: {dataset.data.y.shape if hasattr(dataset.data, 'y') and dataset.data.y is not None else 'None'}")
    print(f"Dataset.data.y unique values: {torch.unique(dataset.data.y) if hasattr(dataset.data, 'y') and dataset.data.y is not None else 'None'}")
