import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train_with_logging as train


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis with Improved MulT Model')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulTImproved',
                    help='name of the model to use (MulTImproved for improved model with modal weight fusion)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')
parser.add_argument('--kernel_size_l', type=int, default=1,
                    help='temporal conv kernel size for text (default: 1)')
parser.add_argument('--kernel_size_v', type=int, default=1,
                    help='temporal conv kernel size for vision (default: 1)')
parser.add_argument('--kernel_size_a', type=int, default=1,
                    help='temporal conv kernel size for audio (default: 1)')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult_improved',
                    help='name of the trial (default: "mult_improved")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

if dataset == 'iemocap':
    args.num_heads = 10
    args.kernel_size_l = 3
    args.kernel_size_v = 3
    args.kernel_size_a = 5
    args.res_dropout = 0.25

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

# Note: set_default_tensor_type is deprecated in PyTorch 2.1+
# The code will work without it as tensors are explicitly moved to device
# torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
# 对于Improved模型，保持首字母大写格式（MULTModelImproved）
hyp_params.model = args.model.strip() if 'Improved' in args.model.strip() else str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


####################################################################
#
# Modified train function to use improved model
#
####################################################################

def initiate_improved(hyp_params, train_loader, valid_loader, test_loader):
    from src import models_improved
    from torch import optim, nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    model = models_improved.MULTModelImproved(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    
    ctc_criterion = None
    ctc_a2l_module, ctc_v2l_module = None, None
    ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train.train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    print("=" * 50)
    print("Using Improved MulT Model with Modal Weight Fusion")
    print("=" * 50)
    test_loss = initiate_improved(hyp_params, train_loader, valid_loader, test_loader)

