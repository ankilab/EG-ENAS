import time
from sklearn.metrics import accuracy_score
import os
import torch
from torch import optim
import torch.nn as nn
from icecream import ic
from helpers import show_time
from datetime import datetime
#from lion_pytorch import Lion
from helpers import Lion
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision.transforms import v2

            

#######################################################################

from utils.train_cfg import (
    get_cfg,
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from utils.distillers import Vanilla
from tqdm import tqdm
from collections import OrderedDict
import os
from torch.optim.lr_scheduler import _LRScheduler


def load_checkpoint(checkpoint_path, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

class BatchWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_batches, last_epoch=-1):
        self.warmup_batches = warmup_batches
        super(BatchWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_batches:
            warmup_factor = (self.last_epoch + 1) / self.warmup_batches
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class TrainerDistillation:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, test=False):
        self.cfg = get_cfg()
        #for key in metadata.keys():
        #    if "train_config_path" in key:
        cfg_path=metadata["train_config_path"]
        #        break
        #    else:
        #        cfg_path=f"{SUBMISSION_PATH}configs/train/vanilla_generation_adam.yaml"
        self.cfg.merge_from_file(cfg_path)
        self.cfg.DATASET.TYPE=metadata["codename"]
        self.cfg.DATASET.CLASSES=metadata["num_classes"]
        self.cfg.DATASET.INPUT_SHAPE=metadata["input_shape"]
        for key in metadata.keys():
            if "experiment_name" in key:
                self.cfg.EXPERIMENT.NAME=metadata["experiment_name"]
        
        self.device=device
        
        self.distiller = Vanilla(model, self.cfg.SOLVER.LABEL_SMOOTHING) #No distillation at the moment
        if torch.cuda.is_available():
            self.distiller= torch.nn.DataParallel(self.distiller.cuda())
        self.train_loader = train_dataloader
        self.val_loader = valid_dataloader
        
        self.optimizer = self.init_optimizer() #Load from cfg. SGD by default.
        if self.cfg.SOLVER.WARMUP==True:
            self.warmup_scheduler = BatchWarmupScheduler(self.optimizer, warmup_batches=int(len(train_dataloader)/2))
        if self.cfg.SOLVER.LR_SCHEDULER=="cosine_annealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(self.cfg.SOLVER.SCHEDULER_EPOCHS, self.cfg.SOLVER.EPOCHS), eta_min=self.cfg.SOLVER.MIN_LR) # CosineAnnealingLR replacing the scheduler defined in the cfg file.
        elif self.cfg.SOLVER.LR_SCHEDULER=="one_cycle":
            print(self.cfg.SOLVER.LR)
            print(len(self.train_loader))
            print(self.cfg.SOLVER.EPOCHS)
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.cfg.SOLVER.LR, steps_per_epoch=len(self.train_loader), epochs=max(self.cfg.SOLVER.SCHEDULER_EPOCHS, self.cfg.SOLVER.EPOCHS))
        elif self.cfg.SOLVER.LR_SCHEDULER=="step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        else:
            raise NotImplementedError(self.cfg.SOLVER.LR_SCHEDULER)
        self.best_acc = -1

        # init loggers
        for key in metadata.keys():
            if "experiment_name" in key:
                self.log_path=metadata["experiment_name"] # Folder to save the training results. Passed in the metadata from NAS file.
                if not os.path.exists(self.log_path):
                    os.makedirs(self.log_path)
                break

       
    def init_optimizer(self):
        if self.cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE=="Lion":
            optimizer= Lion(self.distiller.module.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(), lr=self.cfg.SOLVER.LR, betas=(0.9, 0.99), weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.TYPE=="Adam":
            optimizer=optim.AdamW(self.distiller.module.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(), lr=self.cfg.SOLVER.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise NotImplementedError(self.cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.6f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, return_acc=False ):
        epoch = 1
        start_time=time.time()
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("Total time\t" + "{:.2f}".format(float(time.time()-start_time)))
          # Return trained student
        if return_acc:
            return self.train_acc,self.best_acc, self.epoch_time
        else:
            return self.distiller.module.student if torch.cuda.is_available() else self.distiller.student

    def train_epoch(self, epoch):
        #lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        
        lr = self.optimizer.param_groups[0]['lr']
        
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        #pbar = tqdm(range(num_iter))

        # train loops
        start_epoch_time=time.time()
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
        #    pbar.set_description(log_msg(msg, "TRAIN"))
        #    pbar.update()
            if epoch==1 and self.cfg.SOLVER.WARMUP:
                self.warmup_scheduler.step()
                #print(self.optimizer.param_groups[0]['lr'])
        #pbar.close()
        if (epoch>1) or (self.cfg.SOLVER.WARMUP==False):
            self.scheduler.step()
        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, self.cfg.SOLVER.TOPK)

            

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
                "epoch_time": time.time()-start_epoch_time,

            }
        )
        self.log(lr, epoch, log_dict)
        # saving checkpoint

        if test_acc >= self.best_acc:
            
            student_state = {"model": self.distiller.module.student.state_dict() if torch.cuda.is_available() else self.distiller.student.state_dict() }
            #save_checkpoint(state, os.path.join(self.log_path, "best"))
            self.train_acc=train_meters["top1"].avg
            self.epoch_time=log_dict["epoch_time"]
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, self.cfg.SOLVER.TOPK))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-{}:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            self.cfg.SOLVER.TOPK,
            train_meters["top5"].avg,
        )
        return msg


    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    The prediction function will define how the test dataloader will be passed through your model. It will receive:
        * test_dataloader created by your DataProcessor
    
    And expects as output:
        A list/array of predicted class labels of length=n_test_datapoints, i.e, something like [0, 0, 1, 5, ..., 9] 
    
    See the example submission for how this should look.
    """

    def predict(self, test_loader):
        self.distiller.eval()
        predictions = []
        for data in test_loader:
            data = data.to(self.device)
            output = self.distiller.forward(image=data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions
    
##############################################################

class Trainer(TrainerDistillation):
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, test=False):
        super().__init__(model, device, train_dataloader, valid_dataloader, metadata)
        if test:
            SUBMISSION_PATH="our_submission/"
        else:
            SUBMISSION_PATH=""
        ic(metadata)
        #cfg_path=f"{SUBMISSION_PATH}../configs/train/finetuning_generation_adam.yaml"
        if metadata['input_shape'][1]==1 or metadata['input_shape'][1]==3:
            current_transforms=self.train_loader.dataset.transform.transforms
            self.train_loader.dataset.transform=v2.Compose([v2.RandAugment(magnitude=9)]+[current_transforms[-1]])
            ic(self.train_loader.dataset.transform)
        
        
        cfg_path=metadata["train_config_path"]
        ic(cfg_path)
        
        self.cfg.merge_from_file(cfg_path)
        self.cfg.DATASET.TYPE=metadata["codename"]
        self.cfg.DATASET.CLASSES=metadata["num_classes"]
        self.cfg.DATASET.INPUT_SHAPE=metadata["input_shape"]
        
        for key in metadata.keys():
            if "experiment_name" in key:
                self.cfg.EXPERIMENT.NAME=metadata["experiment_name"]
            else:
                SAVE_PATH=f"{os.getenv('WORK')}/NAS_COMPETITION_RESULTS/full_training_evonas"
                self.cfg.EXPERIMENT.NAME=f"{SAVE_PATH}/finetuning/{metadata['codename']}"
        self.log_path=self.cfg.EXPERIMENT.NAME

        os.makedirs(self.log_path, exist_ok=True)

        ic(self.cfg.SOLVER.LR)
        ic(self.cfg.SOLVER.EPOCHS-self.cfg.SOLVER.SWA_START)
        ic(self.cfg.SOLVER.EPOCHS)
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.distiller.module.student)
        self.swa_start =self.cfg.SOLVER.SWA_START
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.cfg.SOLVER.MIN_LR, anneal_strategy="cos", anneal_epochs=self.cfg.SOLVER.EPOCHS-self.cfg.SOLVER.SWA_START)
        #self.ema_model = torch.optim.swa_utils.AveragedModel(self.distiller.module.student, \
        #             multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))
    
        self.patience = 10
        self.early_stop_counter = 0
    
    def train(self,return_acc=False):
        epoch = 1
        start_time=time.time()
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            self.train_epoch(epoch)
            epoch += 1
        torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, self.device)
        #torch.optim.swa_utils.update_bn(self.train_loader, self.ema_model, self.device)
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("Total time\t" + "{:.2f}".format(float(time.time()-start_time)))
        # Return trained student
        if return_acc:
            return self.train_acc,self.best_acc, self.epoch_time
        else:
            return self.distiller.module.student if torch.cuda.is_available() else self.distiller.student
    def train_epoch(self, epoch):
        #lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        
        lr = self.optimizer.param_groups[0]['lr']
        
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        #pbar = tqdm(range(num_iter))

        # train loops
        start_epoch_time=time.time()
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            #pbar.set_description(log_msg(msg, "TRAIN"))
            #pbar.update()
            if epoch==1 and self.cfg.SOLVER.WARMUP:
                self.warmup_scheduler.step()
                #print(self.optimizer.param_groups[0]['lr'])
            # EMA update logic
            #self.ema_model.update_parameters(self.distiller.module.student)
                
        #pbar.close()
        
        
        # SWA update logic
        if epoch > self.swa_start:
            self.swa_model.update_parameters(self.distiller.module.student)
            self.swa_scheduler.step()
        elif (epoch > 1) or (self.cfg.SOLVER.WARMUP == False):
            self.scheduler.step()


        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, self.cfg.SOLVER.TOPK)
            

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
                "epoch_time": time.time()-start_epoch_time,
            }
        )
        self.log(lr, epoch, log_dict)
        

        if test_acc >= self.best_acc:
        # saving checkpoint
            student_state = {"model": self.distiller.module.student.state_dict() if torch.cuda.is_available() else self.distiller.student.state_dict() }
            self.train_acc=train_meters["top1"].avg
            self.epoch_time=log_dict["epoch_time"]
            save_checkpoint(student_state, os.path.join(self.log_path, "student_best"))            
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if (self.early_stop_counter>=6) and (epoch <= self.swa_start):
                self.swa_model.update_parameters(self.distiller.module.student)
    def predict(self, test_loader, use_swa=True):
        checkpoint_path = os.path.join(self.log_path, "student_best")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distiller.module.student = load_checkpoint(checkpoint_path, self.distiller.module.student, device)
        
        if use_swa:
            distiller= Vanilla(self.swa_model, 0.0)
            if torch.cuda.is_available():
               distiller= torch.nn.DataParallel(distiller.cuda())
        else:
            distiller=self.distiller
        distiller.eval()
        predictions = []
        for data in test_loader:
            data = data.to(self.device)
            output = distiller.forward(image=data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions