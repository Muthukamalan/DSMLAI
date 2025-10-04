import os
import zipfile
from pathlib import Path
import requests
from tqdm.auto import tqdm
import torch
from matplotlib import pyplot as plt


def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

# image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")


def train_step(
        model:torch.nn.Module,
        dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        device:torch.device
)->tuple[float,float]:
    model.train()
    train_loss:float=0.
    train_acc :float=0.
    for batch_idx,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        y_pred:torch.Tensor = model(X)
        loss:torch.Tensor = loss_fn(y_pred,y)
        train_loss +=loss.item()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class==y).sum().item() / len(y_pred)
        optimizer.zero_grad()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



def test_step(
        model:torch.nn.Module,
        dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        device:torch.device
)->tuple[float,float]:
    model.eval()
    test_loss:float=0.
    test_acc :float=0.
    
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            y_pred:torch.Tensor = model(X)
            loss:torch.Tensor = loss_fn(y_pred,y)
            test_loss +=loss.item()
            y_pred_class = y_pred.argmax(dim=1)
            test_acc += (y_pred_class==y).sum().item() / len(y_pred_class)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader,test_dataloader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,loss_fn:torch.nn.Module, epochs:int, device:torch.device)->dict[str,list]:
    results:dict = {
        'train_loss':[],
        'test_loss' :[],
        'train_acc' :[],
        'test_acc'  :[]
    }

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model,train_dataloader,loss_fn,optimizer,device)
        test_loss, test_acc  = test_step(model,test_dataloader,loss_fn,device)

        print(f"Epoch:{epoch+1} | train_loss:{train_loss:4f} | train_acc:{train_acc:4f} | test_loss:{test_loss:4f} | test_acc:{test_acc:4f}")
        
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
    return results



# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def plot_paticfied_image(image:torch.Tensor,label:int,class_names:list[str],patch_size:int=16)->None:
    '''
        Image of shape: H,W,C
        label: int 
        class_names:list[str]
    '''
    H,W,C = image.shape
    number_of_patches:int = H/patch_size 
    assert H%patch_size==0, "Image size must be divisible bypatch_size"
    print(f"Number of patches of row:{number_of_patches}\n Number of patches per column:{number_of_patches}\nTotal Patches: {number_of_patches*number_of_patches}\nPatch size:{patch_size} pixels x {patch_size} pixels")

    fig,axs = plt.subplots(nrows=H//patch_size, ncols= H//patch_size, figsize=(number_of_patches,number_of_patches),sharex=True,sharey=True)
    
    for row, patch_height in enumerate(range(0,H,patch_size)):
        for col, patch_width in enumerate(range(0,W,patch_size)):
            axs[row,col].imshow(image[patch_height:patch_height+patch_size, patch_width:patch_width+patch_size, :])
            
            axs[row,col].set_ylabel( row+1, rotation="horizontal", horizontalalignment="right", verticalalignment="center")

            axs[row,col].set_xlabel(col+1)
            axs[row,col].set_xticks([]); axs[row,col].set_yticks([]); axs[row,col].label_outer()

    fig.suptitle(f"{class_names[label]} -> Patchified",fontsize=16)
    plt.show()