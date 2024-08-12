from yacs.config import CfgNode as CN
import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))

def get_cfg():
    CFG = CN()


    
    # Experiment
    CFG.EXPERIMENT = CN()
    CFG.EXPERIMENT.PROJECT = "NAS"
    CFG.EXPERIMENT.NAME = ""
    #CFG.EXPERIMENT.TAG = "default"
    CFG.EXPERIMENT.LOGIT_STAND = False

    # Dataset
    CFG.DATASET = CN()
    CFG.DATASET.TYPE = ""
    CFG.DATASET.CLASSES = 0
    CFG.DATASET.INPUT_SHAPE = ()
    CFG.DATASET.NUM_WORKERS = 2
    CFG.DATASET.TEST = CN()
    CFG.DATASET.TEST.BATCH_SIZE = 64

    # Distiller
    CFG.DISTILLER = CN()
    CFG.DISTILLER.TYPE = "NONE"  # Vanilla as default
    CFG.DISTILLER.TEACHER = "ResNet50"
    CFG.DISTILLER.STUDENT = "resnet32"

    # Solver
    CFG.SOLVER = CN()
    CFG.SOLVER.TRAINER = "base"
    CFG.SOLVER.BATCH_SIZE = 64
    CFG.SOLVER.EPOCHS = 50
    CFG.SOLVER.LR = 0.05
    CFG.SOLVER.LR_SCHEDULER = "cosine_annealing"
    CFG.SOLVER.SCHEDULER_EPOCHS = 50
    CFG.SOLVER.MIN_LR= 0.0
    CFG.SOLVER.LABEL_SMOOTHING=0.0
    CFG.SOLVER.WARMUP=False
    #CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
    #CFG.SOLVER.LR_DECAY_RATE = 0.1
    CFG.SOLVER.WEIGHT_DECAY = 0.0001
    CFG.SOLVER.MOMENTUM = 0.9
    CFG.SOLVER.TYPE = "SGD"
    CFG.SOLVER.SWA_START= 40
    CFG.SOLVER.TOPK=5

    CFG.KD = CN()
    CFG.KD.TEMPERATURE = 2
    CFG.KD.LOSS = CN()
    CFG.KD.LOSS.CE_WEIGHT = 0.1
    CFG.KD.LOSS.KD_WEIGHT = 9
    CFG.KD.LOSS.KD_EPOCHS= 5
    CFG.KD.LOSS.KD_REDUCTION=True  
   
    # CAT_KD CFG
    #CFG.CAT_KD = CN()
    #CFG.CAT_KD.IF_NORMALIZE = True
    #CFG.CAT_KD.onlyCAT = False
    #CFG.CAT_KD.IF_BINARIZE = False

    #CFG.CAT_KD.IF_OnlyTransferPartialCAMs = False
    #CFG.CAT_KD.CAMs_Nums = 100
    #CFG.CAT_KD.Strategy = 0

    #CFG.CAT_KD.LOSS = CN()
    #CFG.CAT_KD.LOSS.CE_WEIGHT = 1.0
    #CFG.CAT_KD.LOSS.CAT_loss_weight = 400.0
    #CFG.CAT_KD.LOSS.CAM_RESOLUTION = 2

    #CFG.CAT_KD.teacher_dir = None
    #CFG.CAT_KD.student_dir = None

    # DKD(Decoupled Knowledge Distillation) CFG
    #CFG.DKD = CN()
    #CFG.DKD.CE_WEIGHT = 1.0
    #CFG.DKD.ALPHA = 1.0
    #CFG.DKD.BETA = 8.0
    #CFG.DKD.T = 4.0
    #CFG.DKD.WARMUP = 1
    return CFG


##################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller, k=5):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    #pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, k))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-{k}:{top5.avg:.3f}".format(
                top1=top1,k=k, top5=top5
            )
            #pbar.set_description(log_msg(msg, "EVAL"))
            #pbar.update()
    #pbar.close()
    return top1.avg, top5.avg, losses.avg


def validate_npy(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    #pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        start_eval = True
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            output = nn.Softmax()(output)
            if start_eval:
                all_image = image.float().cpu()
                all_output = output.float().cpu()
                all_label = target.float().cpu()
                start_eval = False
            else:
                all_image = torch.cat((all_image, image.float().cpu()), dim=0)
                all_output = torch.cat((all_output, output.float().cpu()), dim=0)
                all_label = torch.cat((all_label, target.float().cpu()), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            #pbar.set_description(log_msg(msg, "EVAL"))
            #pbar.update()
    all_image, all_output, all_label = all_image.numpy(), all_output.numpy(), all_label.numpy()
    #pbar.close()
    return top1.avg, top5.avg, losses.avg, all_image, all_output, all_label


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
