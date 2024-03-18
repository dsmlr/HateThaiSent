import torch.nn as nn
import torch

class MultiTaskLoss(nn.Module):
    def __init__(self, device):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2).to(device), requires_grad=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds_task1, preds_task2, targets_task1, targets_task2):
        
        loss_1 = self.cross_entropy(preds_task1, targets_task1)
        loss_2 = self.cross_entropy(preds_task2, targets_task2)
        precision_1 = torch.exp(-self.log_vars[0])
        precision_2 = torch.exp(-self.log_vars[1])
        loss_1 = precision_1 * loss_1
        loss_2 = precision_2 * loss_2
        
        return loss_1 + loss_2
    
    
def setup_optimizer(model,mtl_loss):
    other_params_value = []
    for name, value in mtl_loss.named_parameters():
        other_params_value.append(value)
    
    optimizer = torch.optim.AdamW([{"params":model.parameters(), "lr":2e-5}, {"params":other_params_value}], lr=1e-3, eps=1e-9)
    return optimizer