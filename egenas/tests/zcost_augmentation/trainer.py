import time
import os
import torch
from torch import optim
import torch.nn as nn
from icecream import ic
from datetime import datetime
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision.transforms import v2
import numpy as np
import torch.nn.functional as F

            

#######################################################################

from train_cfg import (
    get_cfg,
    show_cfg,
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)

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

class Vanilla(nn.Module):
    def __init__(self, student, label_smoothing=0.0):
        super(Vanilla, self).__init__()
        self.student = student
        self.label_smoothing=label_smoothing

    def get_learnable_parameters(self):
        #return [v for k, v in self.student.named_parameters()]
        return [v for k, v in self.student.named_parameters() if v.requires_grad]

    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        loss = F.cross_entropy(input=logits_student, target=target, label_smoothing=self.label_smoothing)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        #return self.student(image)[0]
        return self.student(image)
        
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
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, teachers=[]):
        self.cfg = get_cfg()

        cfg_path=metadata["train_config_path"]
        self.cfg.merge_from_file(cfg_path)
        self.cfg.DATASET.TYPE=metadata["codename"]
        self.cfg.DATASET.CLASSES=metadata["num_classes"]
        self.cfg.DATASET.INPUT_SHAPE=metadata["input_shape"]
        for key in metadata.keys():
            if "experiment_name" in key:
                self.cfg.EXPERIMENT.NAME=metadata["experiment_name"]
        
        self.device=device
        
        self.distiller =  Vanilla(model, self.cfg.SOLVER.LABEL_SMOOTHING)

        ####################################################################################
        if torch.cuda.is_available():
                    self.distiller.to("cuda")
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

        SAVE_PATH=f"full_training"
        if metadata["experiment_name"]=="":
            self.cfg.EXPERIMENT.NAME=f"{SAVE_PATH}/{metadata['codename']}"
        else:
            self.cfg.EXPERIMENT.NAME=f"{metadata['experiment_name']}"
            
        self.log_path=self.cfg.EXPERIMENT.NAME
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


       
    def init_optimizer(self):
        if self.cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE=="Adam":  
            #specific_layers = ["s1.b2","s1.b3","s3", "s2.b2","s2.b3","s2.b4","s2.b5","s2.b6","s2.b7"]

            # Get all parameters of the model
            #all_params = self.distiller.named_parameters()

            # Separate parameters into different groups
            #specific_params = []
            #other_params = []

            #for name, param in all_params:
            #    if any(layer_name in name for layer_name in specific_layers):
            #        print(name)
            #        specific_params.append(param)
            #    else:
            #        other_params.append(param)

            # Define the optimizer with different parameter groups
            #optimizer = optim.AdamW([
            #    {'params': specific_params, 'lr': self.cfg.SOLVER.LR * 0.1},  # Lower learning rate for specific layers
            #    {'params': other_params, 'lr': self.cfg.SOLVER.LR}  # Default learning rate for other layers
            #], betas=(0.9, 0.999), eps=1e-08, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
            optimizer=optim.AdamW(self.distiller.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(), lr=self.cfg.SOLVER.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
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

        ####### Log initial weights performance ######
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, self.cfg.SOLVER.TOPK)
        train_acc, train_acc_top5, train_loss = validate(self.train_loader, self.distiller, self.cfg.SOLVER.TOPK)
        log_dict = OrderedDict(
            {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
                "epoch_time": time.time()-start_time,

            }
        )
        self.log(0.0, 0, log_dict)
        ##############################################

        #for name, param in self.distiller.named_parameters():
        #    param.requires_grad = True
        
        #self.optimizer=optim.AdamW(self.distiller.get_learnable_parameters() if torch.cuda.is_available() else self.distiller.get_learnable_parameters(), lr=self.cfg.SOLVER.LR, betas=(0.9, 0.999),
        #     eps=1e-08, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

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
            return self.distiller.student if torch.cuda.is_available() else self.distiller.student

    def train_epoch(self, epoch):
        #lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        
        lr = self.optimizer.param_groups[0]['lr']
        #ic(lr)
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
            #ic(idx)
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
        #ic(test_acc)
            

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
        #ic(self.log_path)
        #ic(epoch)
        #ic(log_dict)
        self.log(lr, epoch, log_dict)
        # saving checkpoint

        if test_acc >= self.best_acc:
            
            student_state = {"model": self.distiller.student.state_dict() if torch.cuda.is_available() else self.distiller.student.state_dict() }
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
        #ic(acc1)
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-{}:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            self.cfg.SOLVER.TOPK,
            train_meters["top5"].avg,
        )
        #ic(msg)
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

        
        cfg_path=metadata["train_config_path"]
        ic(cfg_path)
        
        self.cfg.merge_from_file(cfg_path)
        self.cfg.DATASET.TYPE=metadata["codename"]
        self.cfg.DATASET.CLASSES=metadata["num_classes"]
        self.cfg.DATASET.INPUT_SHAPE=metadata["input_shape"]
        
        SAVE_PATH=f"full_training"
        if metadata["experiment_name"]=="":
            self.cfg.EXPERIMENT.NAME=f"{SAVE_PATH}/{metadata['codename']}"
        else:
            self.cfg.EXPERIMENT.NAME=f"{metadata['experiment_name']}"
            
        self.log_path=self.cfg.EXPERIMENT.NAME

        os.makedirs(self.log_path, exist_ok=True)

        ic(self.cfg.SOLVER.LR)
        ic(self.cfg.SOLVER.EPOCHS-self.cfg.SOLVER.SWA_START)
        ic(self.cfg.SOLVER.EPOCHS)
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.distiller.student)
        self.swa_start =self.cfg.SOLVER.SWA_START
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.cfg.SOLVER.MIN_LR, anneal_strategy="cos", anneal_epochs=self.cfg.SOLVER.EPOCHS-self.cfg.SOLVER.SWA_START)

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

        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("Total time\t" + "{:.2f}".format(float(time.time()-start_time)))
        # Return trained student
        if return_acc:
            return self.train_acc,self.best_acc, self.epoch_time
        else:
            return self.distiller.student if torch.cuda.is_available() else self.distiller.student
    def train_epoch(self, epoch):
        
        lr = self.optimizer.param_groups[0]['lr']
        
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        
        # train loops
        start_epoch_time=time.time()
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)

            if epoch==1 and self.cfg.SOLVER.WARMUP:
                self.warmup_scheduler.step()
                
        #pbar.close()
        
        
        # SWA update logic
        if epoch > self.swa_start:
            self.swa_model.update_parameters(self.distiller.student)
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
            student_state = {"model": self.distiller.student.state_dict() if torch.cuda.is_available() else self.distiller.student.state_dict() }
            self.train_acc=train_meters["top1"].avg
            self.epoch_time=log_dict["epoch_time"]
            save_checkpoint(student_state, os.path.join(self.log_path, "student_best"))            
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if (self.early_stop_counter>=6) and (epoch <= self.swa_start):
                self.swa_model.update_parameters(self.distiller.student)
                
    def predict(self, test_loader, use_swa=True):
        checkpoint_path = os.path.join(self.log_path, "student_best")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distiller.student = load_checkpoint(checkpoint_path, self.distiller.student, device)
        
        if use_swa:
            distiller= Vanilla(self.swa_model, 0.0)
            if torch.cuda.is_available():
               #distiller= torch.nn.DataParallel(distiller.cuda())
               distiller.to("cuda")
        else:
            distiller=self.distiller
        distiller.eval()
        predictions = []
        for data in test_loader:
            data = data.to(self.device)
            output = distiller.forward(image=data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions