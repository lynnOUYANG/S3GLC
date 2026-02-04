import warnings
import scipy
from tqdm import tqdm
from model.Graph_encoder import to_data_list
warnings.filterwarnings("ignore")
from arguments import arg_parse
from torch_geometric.data import DataLoader
from Utils.aug import TUDataset_aug
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from torch_geometric.data import InMemoryDataset
import numpy as np
import os.path as osp
import os
import shutil
import torch
import torch.nn.functional as F
import random
from itertools import repeat
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from Utils.evaluate_embedding import cluster_acc
from model.model import S3GLC
from loss.losses import *
import scipy
from model.Graph_encoder import to_data_list, generate_graph_encoding
from load_reddit12k import load_reddit12k_dataset
from Utils.preprocess import preprocess_ppa_dataset,preprocess_collab_dataset



def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


if __name__ == '__main__':
    args = arg_parse()
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if args.gpu is not None:
        
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"Using GPU device: cuda:{args.gpu}")
        else:
            print(f"Warning: Specified GPU device {args.gpu} is not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    
    accuracies = {'acc': [], 'nmi': [], 'ari': [], 'randomforest': []}
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    
    dataset_name = args.DS if args.DS else 'ENZYMES'

    for DS in [dataset_name]:#
        args.DS=DS
       
        path = osp.join('./', 'data', DS)
        
        if DS=='PPA':
            dataset = PygGraphPropPredDataset(name="ogbg-ppa", root=path) 
            dataset = preprocess_ppa_dataset(dataset, max_graphs=20000)
        elif DS=='REDDIT-12K':
            dataset = load_reddit12k_dataset('./data/REDDIT-12K')
        elif DS=='COLLAB':
            dataset = TUDataset(path, name=DS)
            dataset = preprocess_collab_dataset(dataset)
        else:
            dataset = TUDataset_aug('path', name=DS, aug=args.aug,cleaned=True)
        
        n_cluster = len(np.unique(dataset.data.y))
        dataset_num_features = max(dataset.data.num_features, 1)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               pin_memory=torch.cuda.is_available())
        precomputed_graph_embeddings = []
        for idx, data in enumerate(dataloader):
            if dataset_name=='PPA' or dataset_name=='REDDIT-12K' or dataset_name=='COLLAB':
                data = data
            else:
                data, _ = data
            data = data.to(device)
            data_list = to_data_list(data, args, device=device)
            graph_embedding = generate_graph_encoding(data_list, args.r, device=device, args=args)
            precomputed_graph_embeddings.append(graph_embedding)

        
        print(f"Pre-computation completed! Stored {len(precomputed_graph_embeddings)} batches of graph embeddings")
        
        args.eval=False
       
        args.cluster_emb = n_cluster

        print('================')
        print('Dataset:', DS)
        print('lr: {}'.format(lr))
        print('epochs: {}'.format(epochs))
        print('batch_size: {}'.format(batch_size))
        print('iter: {}'.format(args.iter))
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('clutering embedding dimension: {}'.format(args.cluster_emb))
        print('================')
        
        mode = 'fd'
        measure = 'JSD'
        par = {
            "num_heads": 1,
            "sk_iter_limit": 20,
            "gamma_bound": 0.05,
            "rho_base": 0.5,
            "rho_upper": 1.0,
            "rho_fix": False,
            "rho_strategy": "sigmoid",
            "label_quality_show": False
        }
        sk_loss = SK_loss(par, sk_type="ppot")
        
        iter = args.iter
        ACCList = np.zeros((iter, 1))
        NMIList = np.zeros((iter, 1))
        ARIList = np.zeros((iter, 1))
        ACC_MEAN = np.zeros((1, 2))
        NMI_MEAN = np.zeros((1, 2))
        ARI_MEAN = np.zeros((1, 2))
        
        
        for it in range(iter):
            
            model = S3GLC(args, dataset_num_features, n_cluster, device, e=args.e).to(device)
            awl = AutomaticWeightedLoss(num_losses=2).to(device)
            optimizer = torch.optim.Adam(list(model.parameters()) + list(awl.parameters()), lr=lr)
            acc = -1.
            nmi = -1.
            ari = -1.
            accmax = -1.
            nmimax = -1.
            arimax = -1.
            
            
            history_loss = []
            pbar = tqdm(range(1, epochs + 1))
            prev_batch_z = {}
            for epoch in pbar:
                
                if epoch ==2:
                    model.eval()
                    emb, y, _ = model.get_results(dataloader, sk_loss, prev_batch_z,precomputed_graph_embeddings=precomputed_graph_embeddings,epoch=epoch)
                    kmeans = KMeans(n_clusters=n_cluster, n_init=100,random_state=42)
                    y_pred = kmeans.fit_predict(emb)
                    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
                    
                if epoch % log_interval == 0 and epoch != 0:
                    model.eval()
                    emb, y, cluster = model.get_results(dataloader, sk_loss, prev_batch_z,precomputed_graph_embeddings=precomputed_graph_embeddings,epoch=epoch)
                    
                    y_pred=cluster.argmax(1) 
                    y = y.flatten()
                    acc = cluster_acc(y, y_pred)
                    nmi = nmi_score(y, y_pred)
                    ari = ari_score(y, y_pred)
                    
                
                    if acc > accmax:
                        accmax = acc
                        
                    if ari > arimax:
                        arimax = ari
                    if nmi > nmimax:
                        nmimax = nmi
                    if ari > arimax:
                        arimax = ari
                # print(accmax,acc,nmi,ari)

                model.train()
                
                for idx, data in enumerate(dataloader):
                    z_prev = prev_batch_z.get(idx)
                    
                    #todo
                    loss_all=0
                    if  dataset_name=='PPA' or dataset_name=='REDDIT-12K' or dataset_name=='COLLAB':
                        data= data
                    else:
                        data, _ = data
                    
                    data=data.to(device)
                    
                    optimizer.zero_grad()
                    
                    graph_embedding = precomputed_graph_embeddings[idx]
                    
                    z, q, loss_components = model(
                        data, z_prev,
                        graph_embedding=graph_embedding,epoch=epoch)
                    prev_batch_z[idx] = z.detach()
                    ot_loss, _ = sk_loss(q) 
                    loss = awl([loss_components['contrastive_loss'], ot_loss])
       
                    loss_all += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                history_loss.append(loss_all)
                
                
            print("ACC:",accmax)
            print("NMI:",nmimax)
            print("ARI:",arimax)
            ACCList[it - 1, :] = accmax
            NMIList[it - 1, :] = nmimax
            ARIList[it - 1, :] = arimax

        ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
        NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
        ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

        
        

        print('ACC:\n' + str(ACC_MEAN))
        print('NMI:\n' + str(NMI_MEAN))
        print('ARI:\n' + str(ARI_MEAN))
        
        os.makedirs('./result', exist_ok=True)
        with open('./result/' + args.DS + '_result.txt', 'a') as f:
            f.write(args.DS + '_Result:' + '\n')
            f.write('r:'+ str(args.r)  + '\n')
            f.write('d:'+ str(args.d)  + '\n')
            f.write('ACC_MEAN:' + str(ACC_MEAN[0][0]*100)+' ('+str(ACC_MEAN[0][1]*100) + ')\n')
            f.write('NMI_MEAN:' + str(NMI_MEAN[0][0]*100)+' ('+str(NMI_MEAN[0][1]*100) + ')\n')
            f.write('ARI_MEAN:' + str(ARI_MEAN[0][0]*100)+' ('+str(ARI_MEAN[0][1]*100) + ')\n')
            
            f.write('\n')
        
        
        
        