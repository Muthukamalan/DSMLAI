import torch
import os
import torchvision

DEVICE:torch.DeviceObjType = torch.device('cuda' if torch.cuda.is_available() else "cpu")
IMAGE_PATH:str             = os.path.join(r"C:\Users\muthu\GitHub\DATA 📁\pizza_steak_sushi")
TRAIN_DIR:str              = os.path.join(IMAGE_PATH,'train')
TEST_DIR:str               = os.path.join(IMAGE_PATH,'test')
NUM_WORKERS:str            = os.cpu_count()-1



IMAGE_SIZE:int             = 224
BATCH_SIZE:int             = 32
PATCH_SIZE:int             = 32#56#8
NUM_TRASFORMER_LAYER:int   = 2    # Fixed in my code  < applied:: Parameter Sharing >
EMBEDDING_DIM:int          = 1024
MLP_SIZE:int               = 64
NUM_HEADS:int              = 16




MANUAL_TRANSFORMS          = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    torchvision.transforms.ToTensor(),
])


'''
| patch_size | number_of_patchs|
|------------|-----------------|
| 2 | 112 |
| 4 | 56 |
| 7 | 32 |
| 8 | 28 |
| 14 | 16 |
| 16 | 14 |
| 28 | 8 |
| 32 | 7 |
| 56 | 4 |
| 112 | 2 |
'''