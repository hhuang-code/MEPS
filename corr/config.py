import argparse
import sys

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='func')

# config for FAUST dataset
faust_parser = subparsers.add_parser('faust')

parser.add_argument('--arch_file', type=str, default='faust_arch.json')

faust_parser.add_argument('--dataset', type=str, default='faust')
faust_parser.add_argument('--dataset_path', type=str, default='./MPI-FAUST/xiang')

faust_parser.add_argument('--max_neigh', type=int, default=6)
faust_parser.add_argument('--num_input_channels', type=int, default=3)
faust_parser.add_argument('--num_classes', type=int, default=6890)
faust_parser.add_argument('--num_points', type=int, default=6890)

faust_parser.add_argument('--batch_size', type=int, default=6)
faust_parser.add_argument('--num_iterations', type=int, default=50000)
faust_parser.add_argument('--learning_rate', type=float, default=0.01)
faust_parser.add_argument('--lr_decay_step', type=int, default=480)
faust_parser.add_argument('--lr_decay_rate', type=float, default=0.5)
faust_parser.add_argument('--lr_clip', type=float, default=0.0001)
faust_parser.add_argument('--weight_decay', type=float, default=0.0001)

faust_parser.add_argument('--log_path', type=str, default='log/faust')

faust_parser.add_argument('--model_path', type=str, default='checkpoint/faust')
faust_parser.add_argument('--model_name', type=str, default='faust_best.pth.tar')

faust_parser.add_argument('--result_path', type=str, default='result/faust')

faust_parser.add_argument('--visual_path', type=str, default='visual/faust')

args = parser.parse_args()

print(args)
