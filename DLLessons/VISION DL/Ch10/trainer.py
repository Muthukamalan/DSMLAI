from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self,model,train_loader, optimizer, criterion, scheduler, device) -> None:
        self.train_losses          = []
        self.train_accuracies      = []
        self.epoch_train_accuracies= []
        self.model                 = model 
        self.train_loader          = train_loader  
        self.optimizer             = optimizer 
        self.criterion             = criterion 
        self.device                = device
        self.scheduler             = scheduler
        self.lr_history            = []

    def train(self,epoch, use_l1=False, lambda_l1=0.01):
        self.model.train()

        correct   = 0
        processed = 0
        train_loss= 0

        pbar=tqdm(self.train_loader)
        for batch_id, (inputs,targets) in enumerate(pbar):
            #Transfer to device
            inputs = inputs.to(self.device)
            targets= targets.to(self.device)

            #Init zero grad
            self.optimizer.zero_grad()

            #Predict
            outputs = self.model(inputs)

            #Calculate loss
            loss = self.criterion(outputs,targets)

            l1=0
            if use_l1:
                for p in self.model.parameters():
                    l1+= p.abs().sum()

            loss += (lambda_l1*l1) 

            self.train_losses.append(loss.item())


            # Backpropagation
            loss.backward()

            # Update
            self.optimizer.step()
            self.scheduler.step()

            pred      = outputs.argmax(dim=1,keepdim=True)
            correct  += pred.eq(targets.view_as(pred)).sum().item()
            processed+= len(inputs)

            caccuracy = 100*correct/processed
            
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            pbar.set_description(desc=f"EPOCH={epoch}| LR={self.optimizer.param_groups[0]['lr']}| LOSS={loss.item():3.2f}| BATCH={batch_id}| ACCURACY={caccuracy:0.3f}")
            self.train_accuracies.append(caccuracy)



        # After all the batches are done, append accuracy for epoch
        self.epoch_train_accuracies.append(caccuracy)
        

        return (
            caccuracy,                          # last epoch training accuracy
            train_loss/len(self.train_loader),  # training loss for loader
        )
