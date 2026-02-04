import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from model.gin import Encoder
from model.Graph_encoder import *


try:
    import torch_sparse
    from torch_sparse import SparseTensor
    TORCH_SPARSE_AVAILABLE = True
except ImportError:
    TORCH_SPARSE_AVAILABLE = False

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss




class S3GLC(nn.Module):
    def __init__(self, args, dataset_num_features, n_label, device, e=1):
        super(S3GLC, self).__init__()
        self.e = e
        self.device = device
        self.d = args.d
        self.n_clusters=n_label
        self.args = args
        self.encoder = encoder(args, dataset_num_features, device, n_label)      
        self.embedding_dim = mi_units = args.hidden_dim * args.num_gc_layers      
        self.z_dim = getattr(args, 'd', 10)
        self.ged_dim = getattr(args, 'r', self.embedding_dim)
        self.joint_embedding_dim = self.z_dim
        self.fuse_proj = nn.Linear(self.z_dim * 2, self.z_dim)
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, self.joint_embedding_dim))      
        torch.nn.init.xavier_uniform_(self.cluster_layer.data)
        self.h_to_z = nn.Linear(self.embedding_dim, self.z_dim)
        self.s_to_z = nn.Linear(self.ged_dim, self.z_dim)
        self.h_to_s = nn.Linear(self.embedding_dim, self.ged_dim)
        self.mse_loss = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    def compute_Y(self, sum_fea, h_enc, graph_embedding):
        
        def compute_L(sum_fea, eps=1e-8):
            N = sum_fea.size(0)
            A = sum_fea @ sum_fea.t()
            A = torch.clamp(A, min=0)
            deg = A.sum(dim=1)
            deg = torch.clamp(deg, min=eps)
            deg_inv_sqrt = deg.pow(-0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            norm_A = D_inv_sqrt @ A @ D_inv_sqrt
            I = torch.eye(N, device=sum_fea.device, dtype=sum_fea.dtype)
            L = I - norm_A
            L = L.T @ L
            return L
        
        weight_s = self.s_to_z.weight.data.T
        weight_h = self.h_to_z.weight.data.T
        L = compute_L(sum_fea)
        U_h = self.update_W(weight_h)
        U_s = self.update_W(weight_s)
        A1 = h_enc.T @ h_enc + U_h
        A2 = graph_embedding.T @ graph_embedding + U_s
        minimize = h_enc @ A1.inverse() @ h_enc.T + graph_embedding @ A2.inverse() @ graph_embedding.T
        I = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        L_new = L + 2 * I - minimize
        e_vals, e_vecs = torch.linalg.eigh(L_new)
        k = self.z_dim
        max_indices = torch.argmax(torch.abs(e_vecs), dim=0)
        col_indices = torch.arange(e_vecs.shape[1], device=e_vecs.device)
        signs = torch.sign(e_vecs[max_indices, col_indices])
        e_vecs = e_vecs * signs.unsqueeze(0)
        Y = e_vecs[:, 1:k+1]
        return Y
        
    def forward(self, data,z_ini, graph_embedding,epoch=4):
        
        x=data.x
        edge_index=data.edge_index
        batch=data.batch
        
        if data.edge_attr is not None:
            edge_attr=data.edge_attr
        else:
            edge_attr=None
        s = None
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(self.device)
        
        h_enc = self.encoder(
            x, edge_index, batch, z_ini, edge_attr=edge_attr)
        
        h_enc=h_enc/torch.norm(h_enc, dim=1, keepdim=True)
        
        for _ in range(self.e):
            projected_h = self.h_to_z(h_enc)
            projected_s = self.s_to_z(graph_embedding)
            sum_fea = projected_s + projected_h
            # Compute Y using the new function
            Y = self.compute_Y(sum_fea, h_enc, graph_embedding)
            # Update weights using Y
            weight_s = self.s_to_z.weight.data.T
            weight_h = self.h_to_z.weight.data.T
            U_h = self.update_W(weight_h)
            U_s = self.update_W(weight_s)
            self.h_to_z.weight.data = (torch.linalg.inv(h_enc.T@h_enc+U_h)@h_enc.T@(Y)).T
            self.s_to_z.weight.data = (torch.linalg.inv(graph_embedding.T@graph_embedding+U_s)@graph_embedding.T@(Y)).T
            
        z=sum_fea
     
        h_contras=self.h_to_s(h_enc)
        contrastive_loss=self.graph_contrastive_train(h_contras,graph_embedding)
        # contrastive_loss=0.
        loss_components = {
            'contrastive_loss': contrastive_loss,
        }
      
        alpha=1.0
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        
       
        return z, q, loss_components
    

    def graph_contrastive_train(self,h,s):
        def get_k_nearest_neighbors(feature_matrix, k=10):
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nbrs.fit(feature_matrix)
            distances, indices = nbrs.kneighbors(feature_matrix)
            return indices
        batch_size=h.shape[0]
        criterion_instance = InstanceLoss(batch_size, 1, 0).to(self.device)
        cl_loss_all=0.
        
        
        indices_h=get_k_nearest_neighbors(s.detach().cpu().numpy())
        x_h_knn=h[indices_h]
        x_h_knn=x_h_knn.reshape(x_h_knn.shape[0] * x_h_knn.shape[1], x_h_knn.shape[2])
        for i in range(10):
            num_i=np.arange(batch_size)*10+i
            cl_loss_all+=(criterion_instance(x_h_knn[num_i], h))/10
        cl_loss_all=cl_loss_all*100
        return cl_loss_all

    def l21_norm(self, mat: torch.Tensor) -> torch.Tensor:
        row_norms = torch.norm(mat, p=2, dim=1)   
        return torch.sum(row_norms)  
    def get_results(self, loader, sk_loss, prev_batch_z=None,precomputed_graph_embeddings=None,epoch=4):
        embedding = []
        cluster = []
        y = []
        if prev_batch_z is None:
            prev_batch_z = {}
        with torch.no_grad():
            for idx,data in enumerate(loader):
                if self.args.DS=='MNIST' or self.args.DS=='COLLAB' or self.args.DS=='PPA' or self.args.DS=='REDDIT-12K':
                    data= data
                else:
                   data, data_aug = data
                data = data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                graph_embedding=precomputed_graph_embeddings[idx]
                
                
                z_prev = prev_batch_z.get(idx)
                z, q, loss_components = self.forward(
                    data, z_prev,
                    graph_embedding=graph_embedding,epoch=epoch)
                prev_batch_z[idx] = z.detach()
                _,p=sk_loss(q)
                embedding.append(z.cpu().numpy())
                cluster.append(p.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        cluster = np.concatenate(cluster, 0)
        y = np.concatenate(y, 0)
        return embedding, y, cluster
    def update_W(self, mat: torch.Tensor) -> torch.Tensor:
        
        if mat.dim() != 2:
            raise ValueError("Input must be a 2D tensor of shape [d, m].")
        row_norms = torch.linalg.norm(mat, dim=1).clamp_min(1e-12)
        diag_vals = row_norms.pow(-0.5)*2
        d = mat.size(0)
        indices = torch.arange(d, device=mat.device).repeat(2, 1)
        return torch.sparse_coo_tensor(indices, diag_vals, size=(d, d), device=mat.device)
    





class encoder(nn.Module):
    def __init__(self, args, dataset_num_features, device, n_label=None):
        super(encoder, self).__init__()
        
        self.device = device
        self.args = args  
        self.embedding_dim = mi_units = args.hidden_dim * args.num_gc_layers
        self.embedding_dim_project=int(self.embedding_dim/2)
        self.cluster_embedding = Cluster(self.embedding_dim_project, args.cluster_emb)
        
        self.encoder = Encoder(dataset_num_features, args.hidden_dim, args.num_gc_layers, self.device,use_weight=False)
        # Set d from args if available
        self.z_topk = getattr(args, 'd', None)
 
        self.local_d = FF(self.embedding_dim)
       

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, z_ini, edge_attr=None):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(self.device)      
        y, M = self.encoder(x, edge_index, batch,edge_attr=edge_attr)

        h_enc = self.local_d(y)

        return h_enc

    def get_results(self, loader):
        embedding = []
        y = []

        with torch.no_grad():
            for data in loader:
                if self.args.DS=='MNIST' or self.args.DS=='CIFAR10' or self.args.DS=='CLUSTER':
                    data= data
                else:
                   data, data_aug = data
                data = data.to(self.device)
                data_list=to_data_list(data, self.args, device=self.device)
                graph_embedding = generate_graph_encoding(data_list, self.args.r, device=self.device, args=self.args)
                # data_aug = data_aug.to(device)
                x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                z = self.forward(x, edge_index, batch, num_graphs,graph_embedding=graph_embedding)
                embedding.append(z.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        y = np.concatenate(y, 0)
        return embedding, y

   


class Cluster(nn.Module):
    def __init__(self, input_dim, cluster_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/2), cluster_dim),
            nn.LeakyReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, cluster_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)