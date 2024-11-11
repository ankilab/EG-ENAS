import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Distiller

###################### KD distillation ###################################    

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher, True)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.kd_epochs=cfg.KD.LOSS.KD_EPOCHS
        self.kd_reduction=cfg.KD.LOSS.KD_REDUCTION
        self.label_smoothing=cfg.SOLVER.LABEL_SMOOTHING



    def forward_train(self, image, target, **kwargs):
        #logits_student, _ = self.student(image)
        logits_student = self.student(image)
        if kwargs['epoch']<=self.kd_epochs:
            logits_teachers=[]
            with torch.no_grad():
                for teacher in self.teacher:
                    logits_teacher= teacher(image)
                    #logits_teacher, _ = teacher(image)
                    logits_teachers.append(logits_teacher)

        # losses
        
        
        if kwargs['epoch']<=self.kd_epochs:
            if self.kd_reduction==1:
                kd_loss_w=self.kd_loss_weight*(1-(kwargs['epoch']-1)/self.kd_epochs)
                #kd_loss_w=self.kd_loss_weight/self.kd_epochs
                #loss_ce = kwargs['epoch']*self.ce_loss_weight * F.cross_entropy(logits_student, target)
                loss_ce = self.ce_loss_weight * F.cross_entropy(input=logits_student, target=target, label_smoothing=self.label_smoothing)
                
                loss_kd=0
                for i in range(len(self.teacher)):
                
                    loss_kd =loss_kd+ (kd_loss_w) * kd_loss(
                        logits_student, logits_teachers[i], self.temperature, self.logit_stand
                    )

            else:
                loss_ce = self.ce_loss_weight * F.cross_entropy(input=logits_student, target=target, label_smoothing=self.label_smoothing)
                loss_kd=0
                for i in range(len(self.teacher)):
                    loss_kd =loss_kd+ (self.kd_loss_weight) * kd_loss(
                        logits_student, logits_teachers[i], self.temperature, self.logit_stand
                    )
            losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
                }
        else:
            loss_ce =F.cross_entropy(input=logits_student, target=target, label_smoothing=self.label_smoothing)
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_ce,
            }
        return logits_student, losses_dict

    def forward_test(self, image):
        return self.student(image)

    def get_learnable_parameters(self):
        #return [v for k, v in self.student.named_parameters()]
        return [v for k, v in self.student.named_parameters() if v.requires_grad]  

