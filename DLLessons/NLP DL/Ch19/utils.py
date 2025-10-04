import torch
class AvgMeter:
    def __init__(self,name="Metric") -> None:
        self.name = name
        self.reset()

    def reset(self):
        self.avg:int=0
        self.sum:int=0
        self.count:int=0

    def update(self,val,count=1):
        self.count += count
        self.sum += val*count
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
    

def get_lr(optimizer:torch.optim.Optimizer)->int:
    for param_group in optimizer.param_groups: 
        return param_group['lr']