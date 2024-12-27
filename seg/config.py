import argparse
import sys

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='func')

# config for ShapeNet dataset
shapenet_parser = subparsers.add_parser('shapenet')

shapenet_parser.add_argument('--arch_file', type=str, default='shapenet_arch.json')

shapenet_parser.add_argument('--dataset', type=str, default='shapenet')
shapenet_parser.add_argument('--dataset_path', type=str, default='./shapenet_part_seg_hdf5_data')

shapenet_parser.add_argument('--max_neigh', type=int, default=16)
shapenet_parser.add_argument('--num_input_channels', type=int, default=3)
shapenet_parser.add_argument('--num_categories', type=int, default=16, help='# of shape categories')
shapenet_parser.add_argument('--num_classes', type=int, default=50)
shapenet_parser.add_argument('--num_points', type=int, default=2048)

shapenet_parser.add_argument('--batch_size', type=int, default=48)
shapenet_parser.add_argument('--max_epoch', type=int, default=1000)
shapenet_parser.add_argument('--learning_rate', type=float, default=0.001)
shapenet_parser.add_argument('--lr_decay_step', type=int, default=50)
shapenet_parser.add_argument('--lr_decay_rate', type=float, default=0.7)
shapenet_parser.add_argument('--lr_clip', type=float, default=1e-5)
shapenet_parser.add_argument('--weight_decay', type=float, default=0.00001)

shapenet_parser.add_argument('--log_path', type=str, default='log/shapenet')
shapenet_parser.add_argument('--log_name', type=str, default='shapenet.log')

shapenet_parser.add_argument('--model_path', type=str, default='checkpoint/shapenet')
shapenet_parser.add_argument('--model_name', type=str, default='shapenet.pth.tar')
shapenet_parser.add_argument('--resume_model_name', type=str, default='')

shapenet_parser.add_argument('--result_path', type=str, default='result/shapenet')

shapenet_parser.add_argument('--visual_path', type=str, default='visual/shapenet')

shapenet_parser.add_argument('--visualize', action='store_true')

args = parser.parse_args()

print(args)
