import os
import sys
import time
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils

from adaptive_augmentor import AdaAug
from networks import get_model
from networks.projection import Projection
from dataset import get_num_class, get_dataloaders, get_label_name, get_dataset_dimension
from config import get_warmup_config
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--use_parallel', action='store_true', default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="model name")
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--search_dataset', type=str, default='./', help='search dataset name')
parser.add_argument('--gf_model_name', type=str, default='./', help='gf_model name')
parser.add_argument('--gf_model_path', type=str, default='./', help='gf_model path')
parser.add_argument('--h_model_path', type=str, default='./', help='h_model path')
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--delta', type=float, default=0.3, help="degree of perturbation in magnitude")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of additional hidden layer in augmentation policy projection")
parser.add_argument('--n_proj_hidden', type=int, default=128, help="number of hidden units in augmentation policy projection layers")
parser.add_argument('--restore_path', type=str, default='./', help='restore model path')
parser.add_argument('--restore', action='store_true', default=False, help='restore model default False')

args = parser.parse_args()
debug = True if args.save == "debug" else False
args.save = '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save)
if debug:
    args.save = os.path.join('debug', args.save)
else:
    args.save = os.path.join('eval', args.dataset, args.save)
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    utils.reproducibility(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    #  dataset settings
    n_class = get_num_class(args.dataset)
    class2label = get_label_name(args.dataset, args.dataroot)
    train_queue, valid_queue, _, test_queue = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True)

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')

    #  task model settings
    task_model = get_model(model_name=args.model_name,
                            num_class=n_class,
                            use_cuda=True, data_parallel=False)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(task_model))

    #  task optimization settings
    optimizer = torch.optim.SGD(
        task_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    m, e = get_warmup_config(args.dataset)
    scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=m,
            total_epoch=e,
            after_scheduler=scheduler)
    logging.info(f'Optimizer: SGD, scheduler: CosineAnnealing, warmup: {m}/{e}')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    #  restore setting
    if args.restore:
        trained_epoch = utils.restore_ckpt(task_model, optimizer, scheduler, args.restore_path, location=args.gpu) + 1
        n_epoch = args.epochs - trained_epoch
        logging.info(f'Restoring model from {args.restore_path}, starting from epoch {trained_epoch}')
    else:
        trained_epoch = 0
        n_epoch = args.epochs

    #  load trained adaaug sub models
    search_n_class = get_num_class(args.search_dataset)
    gf_model = get_model(model_name=args.gf_model_name,
                            num_class=search_n_class,
                            use_cuda=True, data_parallel=False)

    h_model = Projection(in_features=gf_model.fc.in_features,
                            n_layers=args.n_proj_layer,
                            n_hidden=args.n_proj_hidden).cuda()

    utils.load_model(gf_model, f'{args.gf_model_path}/gf_weights.pt', location=args.gpu)
    utils.load_model(h_model, f'{args.h_model_path}/h_weights.pt', location=args.gpu)

    for param in gf_model.parameters():
        param.requires_grad = False

    for param in h_model.parameters():
        param.requires_grad = False

    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': args.k_ops,
                    'delta': args.delta,
                    'temp': args.temperature,
                    'search_d': get_dataset_dimension(args.search_dataset),
                    'target_d': get_dataset_dimension(args.dataset)}

    adaaug = AdaAug(after_transforms=after_transforms,
                    n_class=search_n_class,
                    gf_model=gf_model,
                    h_model=h_model,
                    save_dir=args.save,
                    config=adaaug_config)

    #  start training
    for i_epoch in range(n_epoch):
        epoch = trained_epoch + i_epoch
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train(
            train_queue, task_model, criterion, optimizer, epoch, args.grad_clip, adaaug)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj, _, _ = infer(valid_queue, task_model, criterion)
        logging.info('valid_acc %f', valid_acc)

        scheduler.step()

        if epoch % args.report_freq == 0:
            test_acc, test_obj, test_acc5, _ = infer(test_queue, task_model, criterion)
            logging.info('test_acc %f %f', test_acc, test_acc5)

        utils.save_ckpt(task_model, optimizer, scheduler, epoch,
            os.path.join(args.save, 'weights.pt'))

    adaaug.save_history(class2label)
    figure = adaaug.plot_history()
    test_acc, test_obj, test_acc5, _ = infer(test_queue, task_model, criterion)

    logging.info('test_acc %f %f', test_acc, test_acc5)
    logging.info(f'save to {args.save}')


def train(train_queue, model, criterion, optimizer, epoch, grad_clip, adaaug):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)

        #  get augmented training data from adaaug
        aug_images = adaaug(input, mode='exploit')
        model.train()
        optimizer.zero_grad()
        logits = model(aug_images)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.detach().item(), n)
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)

        global_step = step + epoch * len(train_queue)
        if global_step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', global_step, objs.avg, top1.avg, top5.avg)

        # log the policy
        if step == 0:
            adaaug.add_history(input, target)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for input, target in valid_queue:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.detach().item(), n)
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)

    return top1.avg, objs.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
