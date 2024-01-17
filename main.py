import torch
import argparse

from explorer.exp import exp_base

parser = argparse.ArgumentParser(description = "Time Series Forecasting by ZeroChord")

# Simulation setting
parser.add_argument("--test_id", type = str, required = True, help = "test ID")
parser.add_argument("--checkpoints", type = str, default = "./checkpoints/", help = "location to save model checkpoint")

# GPU setting
parser.add_argument("--use_gpu", type = bool, default = True, help = "flag for using GPU")
parser.add_argument("--gpu", type = int, default = 0, help = "device index")
parser.add_argument("--use_multi_gpu", type = bool, default = False, help = "flag for using multiple GPUs")
parser.add_argument("--devices", type = str, default = "0,1", help = "device indices of multiple GPUs")

# Dataloader setting
parser.add_argument("--data_path", type = str, default = "./data/ETTm1.csv", help = "path for raw data")
parser.add_argument("--scale", type = bool, default = False, help = "flag for data scaling [True: scaling, False: not scaling]")
parser.add_argument("--num_workers", type = int, default = 0, help = "num workers for dataloader")

# Forecasting setting
parser.add_argument("--len_seq", type = int, default = 96, help = "input sequence length")
parser.add_argument("--len_pred", type = int, default = 48, help = "prediction sequence length")
parser.add_argument("--len_label", type = int, default = 48, help = "start token length")

# Model setting
parser.add_argument("--model", type = str, default = "DLinear", help = "model name, options: [DLinear]")
parser.add_argument("--batch_size", type = int, default = 8, help = "batch size for dataloader")
parser.add_argument("--individual", action = "store_true", default = False, help = "DLinear: a linear layer for each variate (channel) individually")
parser.add_argument('--enc_input_size', type=int, default=7, help='encoder input size')

# Train setting
parser.add_argument("--optimizer", type = str, default = "Adam", help = "optimizer for model training, options: [Adam]")
parser.add_argument("--learning_rate", type = float, default = 0.0001, help = "learning rate for model training")
parser.add_argument("--loss", type = str, default = "mse", help = "loss function for model training")
parser.add_argument("--only_test", type = bool, default = False, help = "flag for model training [True: train and test, False: only test]")
parser.add_argument("--only_train", type = bool, default = False, help = "flag indicating whether the validation and test are proceeding")
parser.add_argument("--patience", type = int, default = 5, help = "patience for early stopping")
parser.add_argument("--epochs", type = int, default = 5, help = "epochs for model training")

args = parser.parse_args()

# Check if GPU is available
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# Set multi devices if multi gpu is available
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print("Arguments in simulation:", args)

Explorer = exp_base(args)

if not args.only_test: #!#
    Explorer.train()