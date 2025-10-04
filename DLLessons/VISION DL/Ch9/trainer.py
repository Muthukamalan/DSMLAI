from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self,model,train_loader, optimizer, criterion, device) -> None:
        self.train_losses          = []
        self.train_accuracies      = []
        self.epoch_train_accuracies= []
        self.model                 = model 
        self.train_loader          = train_loader  
        self.optimizer             = optimizer 
        self.criterion             = criterion 
        self.device                = device
        self.lr_history            = []

    def train(self,epoch, use_l1=False, lambda_l1=0.01):
        self.model.train()

        lr_trend  = []
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

            pred      = outputs.argmax(dim=1,keepdim=True)
            correct  += pred.eq(targets.view_as(pred)).sum().item()
            processed+= len(inputs)

            if type(self.optimizer)==ReduceLROnPlateau:
                clr = self.optimizer.get_last_lr()
            else:
                clr = self.optimizer.param_groups[0]['lr']
            caccuracy = 100*correct/processed
            lr_trend.append(clr)
            
            pbar.set_description(desc=f"EPOCH={epoch}| LR={clr:3f}| LOSS={loss.item():3.2f}| BATCH={batch_id}| ACCURACY={caccuracy:0.3f}")
            self.train_accuracies.append(caccuracy)


        # After all the batches are done, append accuracy for epoch
            self.epoch_train_accuracies.append(caccuracy)
        
        self.lr_history.extend(lr_trend)

        return (
            caccuracy,                          # last epoch training accuracy
            train_loss/len(self.train_loader),  # training loss for loader
            lr_trend                            # lrs
        )
