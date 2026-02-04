import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DCGLC')
    
    # Dataset parameters
    parser.add_argument('--DS', dest='DS', default='ENZYMES', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)
    
    # Model parameters
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=64,
                        help='')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.0,
                        help='Alpha parameter')
    parser.add_argument('--beta', dest='beta', type=float, default=10.0,
                        help='Beta parameter')
    parser.add_argument('--lamda', dest='lamda', type=float, default=1.0,
                        help='Lambda parameter')
    parser.add_argument('--eta', dest='eta', type=float, default=2.0,
                        help='Eta parameter')
    parser.add_argument('--poe-beta', dest='poe_beta', type=float, default=1.0,
                        help='KL divergence weight used in POE VAE loss')
    parser.add_argument('--poe-tc-ratio', dest='poe_tc_ratio', type=float, default=0.3,
                        help='Total correlation ratio (alpha) used in POE VAE loss')
    parser.add_argument('--poe-style-dim', dest='poe_style_dim', type=int, default=None,
                        help='Style latent dimension for POE VAE (defaults to embedding_dim/2)')
    parser.add_argument('--poe-class-dim', dest='poe_class_dim', type=int, default=None,
                        help='Content latent dimension for POE VAE (defaults to embedding_dim)')
    
    # Training parameters
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--iter', dest='iter', type=int, default=5,
                        help='Number of iterations')
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=1,
                        help='Log interval')
    
    # Device parameters
    parser.add_argument('--gpu', dest='gpu', type=int, default=None,
                        help='GPU device number')
    
    # Augmentation parameters
    parser.add_argument('--aug', dest='aug', default='none',
                        help='Augmentation method')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed')
    
    # Multi-view parameters
    parser.add_argument('--view-number', dest='viewNumber', type=int, default=2,
                        help='Number of views for multi-view learning')
    parser.add_argument('--share', dest='share', action='store_true', default=True,
                        help='Whether to share encoder across views')
    parser.add_argument('--no-share', dest='share', action='store_false',
                        help='Disable sharing encoder across views')
    
    # Debug parameters
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='Enable debug output for loss values and model internals')
    
    # Evaluation parameters
    parser.add_argument('--eval', dest='eval', action='store_const', const=True, default=False)
    parser.add_argument('--mode', dest='mode', default='fast', help='Mode')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.1, help='Gamma parameter')
    
    # Other parameters
    parser.add_argument('--nu', dest='nu', type=float, default=0.1, help='Nu parameter')
    parser.add_argument('--r', dest='r', type=int, default=10,
                        help='Graph Encoding Dimension - dimension of graph encoding embedding (default: 10)')
    parser.add_argument('--d', dest='d', type=int, default=None,
                        help='Number of leading eigenvectors to use when fusing views (default: use all)')
    parser.add_argument('--e', dest='e', type=int, default=1,
                        help='Parameter e (default: 1)')
    
    # Visualization parameters
    parser.add_argument('--vis', dest='vis', action='store_true', default=False,
                        help='Enable t-SNE visualization of clustering results (saves embeddings from best ACC epoch)')
    
    return parser.parse_args()