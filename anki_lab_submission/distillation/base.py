import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher, multi=False):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.multi = multi

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.multi:
            for i in range(len(self.teacher)):
                self.teacher[i]=self.teacher[i].eval()
        else:
            self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student, label_smoothing=0.0):
        super(Vanilla, self).__init__()
        self.student = student
        self.label_smoothing=label_smoothing

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        #logits_student, _ = self.student(image)
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

