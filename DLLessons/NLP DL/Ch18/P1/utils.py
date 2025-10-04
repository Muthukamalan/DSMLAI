import torch
from matplotlib import pyplot as plt 

class DiceLoss(torch.nn.Module):
    '''
    Dice Loss =  1 -  { (2 * intersection + smooth) / (sum of squares of prediction + sum of squares of ground truth + smooth)}
    '''
    def __init__(self,eps=1e-4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self,y_pred:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        # flatten y_pred and ohls
        y_pred:torch.Tensor = y_pred.view(-1)
        y:torch.Tensor      = y.view(-1)

        intersection = torch.sum(y_pred*y)
        union        = y_pred.sum() + y.sum()
        loss         = ( (2.* (intersection+self.eps) )  / (union+self.eps)  )
        return 1-loss





def plot_unet_results(model, datamodule, num_images=5):
    fig, ax = plt.subplots(num_images, 3, figsize=(15, 15))
    # fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(num_images):
        x = datamodule.val_dataset[i][0]
        target = datamodule.val_dataset[i][2]
        predicted = model(x.unsqueeze(0))
        print(x.shape,)
        ax[i][0].imshow(x.permute(1, 2, 0))
        ax[i][0].set_title(f"Input Image-{i+1}")

        ax[i][1].imshow(target.permute(1, 2, 0)[:,:,0])
        ax[i][1].set_title(f"Target Image-{i+1}")

        ax[i][2].imshow(predicted.detach().squeeze(0).permute(1, 2, 0))
        ax[i][2].set_title(f"Predicted Image-{i+1}")

        for a in ax[i]:
            a.set_xticklabels([])
            a.set_yticklabels([])
    plt.show()



def plot_unet_samples(datamodule, num_images=5): 
    fig, ax = plt.subplots(num_images, 4, figsize=(10, 15))
    for i in range(num_images):
        x = datamodule.val_dataset[i][0]
        target = datamodule.val_dataset[i][2]

        ax[i][0].imshow(x.permute(1, 2, 0))
        ax[i][0].set_title(f"Input Image-{i+1}")

        ax[i][1].imshow(target.permute(1, 2, 0)[:,:,0])
        ax[i][1].set_title(f"Target Image Channel::1-{i+1}",fontsize = 8.0)

        ax[i][2].imshow(target.permute(1, 2, 0)[:,:,1])
        ax[i][2].set_title(f"Target Image Channel::2-{i+1}",fontsize = 8.0)

        ax[i][3].imshow(target.permute(1, 2, 0)[:,:,2])
        ax[i][3].set_title(f"Target Image Channel::3-{i+1}",fontsize = 8.0)

        for a in ax[i]:
            a.set_xticklabels([])
            a.set_yticklabels([])
    plt.show()
