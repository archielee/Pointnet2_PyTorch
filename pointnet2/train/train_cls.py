from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse
import numpy as np

from pointnet2.models import Pointnet2ClsMSG as Pointnet
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import CustomDataset
import pointnet2.data.data_utils as d_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=750, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-2,
                        help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=2e5, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("-data_dir", type=str, default="maps")

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )

    num_envs = 100
    num_views = 100
    dataset = CustomDataset(args.data_dir, num_envs, num_views, 750)

    # Create data indices for training and validation splits
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=4, sampler=test_sampler)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              num_workers=4, sampler=train_sampler)

    model = Pointnet(input_channels=1, num_classes=8, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lbmd(it): return max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )

    def bn_lbmd(it): return max(
        args.bn_momentum
        * args.bnm_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    lr_scheduler = lr_sched.LambdaLR(
        optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    if args.visdom:
        viz = pt_utils.VisdomViz(port=args.visdom_port)
    else:
        viz = pt_utils.CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/pointnet2_cls",
        best_name="checkpoints/pointnet2_cls_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz,
    )

    trainer.train(
        it, start_epoch, args.epochs, train_loader, test_loader, best_loss=best_loss
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)
