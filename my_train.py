from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist


from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torch.nn as nn
import cfg
import models_search
import datasets
from functions import train, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger
# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception
import warnings
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed
import os

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random
warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING=1
def setup(local_rank):
    print(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')


def main(args):
    local_rank = args.local_rank
    setup(local_rank)
    setup_seed(args.seed + local_rank)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        #         elif classname.find('Linear') != -1:
        #             if args.init_type == 'normal':
        #                 nn.init.normal_(m.weight.data, 0.0, 0.02)
        #             elif args.init_type == 'orth':
        #                 nn.init.orthogonal_(m.weight.data)
        #             elif args.init_type == 'xavier_uniform':
        #                 nn.init.xavier_uniform(m.weight.data, 1.)
        #             else:
        #                 raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # import network

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:

        gen_net = eval('models_search.' + args.gen_model + '.Generator')(args=args).to(local_rank)
        dis_net = eval('models_search.' + args.dis_model + '.Discriminator')(args=args).to(local_rank)
        gen_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_net)
        dis_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_net)
        gen_net.apply(weights_init)
        dis_net.apply(weights_init)
        gen_net = DDP(gen_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        dis_net = DDP(dis_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have

        args.dis_batch_size = int(args.dis_batch_size / dist.get_world_size())
        args.gen_batch_size = int(args.gen_batch_size / dist.get_world_size())
        args.batch_size = args.dis_batch_size

        # args.num_workers = int((args.num_workers + dist.get_world_size() - 1) / dist.get_world_size())




    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.fid_stat is not None:
        fid_stat = args.fid_stat
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    dataset = datasets.ImageDataset(args, cur_img_size=8)
    train_loader = dataset.train
    train_sampler = dataset.train_sampler
    print(len(train_loader)) if dist.get_rank()==0 else 0
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        #         avg_gen_net = deepcopy(gen_net)
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        #         del avg_gen_net
        #         gen_avg_param = list(p.cuda().to(f"cuda:{args.gpu}") for p in gen_avg_param)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if local_rank== 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if local_rank== 0 else None
        del checkpoint
    else:
        # create new log dir
        assert args.exp_name
        if local_rank== 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])

    if args.local_rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    epoch_bar = tqdm(range(int(start_epoch), int(args.max_epoch))) if dist.get_rank()==0 else range(int(start_epoch), int(args.max_epoch))
    for epoch in epoch_bar:
        train_sampler.set_epoch(epoch)
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if local_rank== 0 else 0
        print(f"path: {args.path_helper['prefix']}") if local_rank== 0 else 0
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              fixed_z,
              lr_schedulers)

        if local_rank == 0 and args.show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch) - 1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            inception_score, fid_score = validate(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            if local_rank== 0:
                logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
            load_params(gen_net, backup_param, args)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False
        # is_best = True
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and local_rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_model': args.gen_model,
                'dis_model': args.dis_model,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': best_fid,
                # 'path_helper': args.path_helper,
                'fixed_z': fixed_z
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        dist.barrier()
        del avg_gen_net

def setup_seed(seed):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = cfg.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    main(args)