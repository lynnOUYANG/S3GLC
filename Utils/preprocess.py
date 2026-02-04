import os
import shutil
import torch
from torch_geometric.data import InMemoryDataset


def preprocess_collab_labels(path, dataset_name: str = "COLLAB") -> None:
    """
    Preprocess COLLAB dataset labels file to convert float labels to integers.
    This fixes the ValueError: invalid literal for int() with base 10: '1.0' error.
    """

    # Try multiple possible paths where the labels file might be
    possible_paths = [
        os.path.join(path, dataset_name, "raw", f"{dataset_name}_graph_labels.txt"),
        os.path.join(path, dataset_name, dataset_name, "raw", f"{dataset_name}_graph_labels.txt"),
        os.path.join(path, dataset_name, dataset_name, "raw_cleaned", f"{dataset_name}_graph_labels.txt"),
        os.path.join(path, dataset_name, dataset_name, dataset_name, f"{dataset_name}_graph_labels.txt"),
    ]

    labels_file = None
    for p in possible_paths:
        if os.path.exists(p):
            labels_file = p
            break

    if labels_file and os.path.exists(labels_file):
        # Read the file
        with open(labels_file, "r") as f:
            lines = f.readlines()

        # Check if conversion is needed (check first few lines)
        needs_conversion = False
        for line in lines[:10]:
            line = line.strip()
            if line and "." in line:
                try:
                    float(line)
                    needs_conversion = True
                    break
                except ValueError:
                    pass

        if needs_conversion:
            # Backup original file
            backup_file = labels_file + ".backup"
            if not os.path.exists(backup_file):
                shutil.copy2(labels_file, backup_file)

            # Convert float labels to integers
            converted_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        # Convert float to int
                        label = int(float(line))
                        converted_lines.append(str(label) + "\n")
                    except ValueError:
                        converted_lines.append(line + "\n")
                else:
                    converted_lines.append("\n")

            # Write converted labels back
            with open(labels_file, "w") as f:
                f.writelines(converted_lines)

            print(f"Converted float labels to integers in {labels_file}")
        else:
            print(
                f"Labels file {labels_file} already contains integer labels, "
                f"skipping conversion"
            )
    else:
        # If file not found, we silently skip; dataset loading code will handle errors if any
        return


def preprocess_ppa_dataset(dataset, max_graphs: int = 20000):
    """
    Preprocess PPA dataset:
    1. Keep only the first max_graphs graphs
    2. Compute node features x from edge_attr (average of incident edges)
    3. Return a new InMemoryDataset
    """
    subset_len = min(len(dataset), max_graphs)
    data_list = []

    for i in range(subset_len):
        data = dataset[i]
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = int(data.num_nodes)

        x = torch.zeros(num_nodes, edge_attr.size(1), device=edge_attr.device)
        deg = torch.zeros(num_nodes, device=edge_attr.device)

        src, dst = edge_index
        x.index_add_(0, src, edge_attr)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        x.index_add_(0, dst, edge_attr)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

        deg = deg.clamp_min(1.0).unsqueeze(1)
        x = x / deg

        data.x = x
        data_list.append(data)

    new_dataset = InMemoryDataset()
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)
    return new_dataset


def preprocess_collab_dataset(dataset):
    """
    Preprocess COLLAB dataset by adding x attribute (all-ones vector) to each graph.
    x shape: [num_nodes, 1]
    """
    data_list = []

    for i in range(len(dataset)):
        data = dataset[i]
        num_nodes = int(data.num_nodes)

        # Create all-ones feature vector with shape [num_nodes, 1]
        x = torch.ones(num_nodes, 1, dtype=torch.float32)

        data.x = x
        data_list.append(data)

    new_dataset = InMemoryDataset()
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)
    return new_dataset


