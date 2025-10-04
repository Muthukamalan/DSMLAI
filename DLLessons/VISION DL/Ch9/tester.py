from tqdm import tqdm 
import torch
from torch import Tensor


class Tester:
    def __init__(self,model,train_loader, criterion, device) -> None:
        self.test_losses          = []
        self.test_accuracies      = []
        self.model                = model 
        self.test_loader          = train_loader  
        self.criterion            = criterion 
        self.device               = device

        self.model = self.model.to(device)

    def test(self):
        self.model.eval()
        correct   = 0
        test_loss= 0


        with torch.no_grad():
            pbar=tqdm(self.test_loader)
            for batch_id, (inputs,targets) in enumerate(pbar):
                #Transfer to device
                inputs = inputs.to(self.device)
                targets= targets.to(self.device)

                #Predict
                outputs = self.model(inputs)

                #Calculate loss
                loss = self.criterion(outputs,targets)
                test_loss += loss.item()

                pred =  outputs.argmax(dim=1,keepdim=True) #get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()


                caccuracy = 100*correct/len(self.test_loader.dataset)
                pbar.set_description(desc=f"LOSS={test_loss:3.2f}| BATCH={batch_id}| ACCURACY={caccuracy:0.3f}")

            test_loss /= len(self.test_loader.dataset)
            self.test_losses.append(test_loss)
       
        self.test_accuracies.append(caccuracy)
        return (
            caccuracy,                          # last epoch training accuracy
            test_loss                           # last epoch loss
        )


    def get_misclassified_images(self):
        self.model.eval()
        images       = []
        predicitions = []
        labels       = []

        with torch.no_grad():
            for data,target in self.test_loader:
                data,target = data.to(self.device) , target.to(self.device)

                outputs = self.model(data)
                _,preds = torch.max(outputs,1)

                for i in range(len(preds)):
                    if preds[i]!=target[i]:
                        images.append(data[i])
                        predicitions.append(preds[i])
                        labels.append(target[i])

        return images,predicitions,labels
