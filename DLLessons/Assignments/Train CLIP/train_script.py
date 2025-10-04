import lightning as pl 
import matplotlib.pyplot as plt
import numpy as np

from lightning.pytorch import loggers as pl_loggers
from functorch.compile import compiled_function,draw_graph
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import torch
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium' )
import os 
from src.utils import *
from src.datamodule import ClipDataModule
from src.model import CLIPModel




dm = ClipDataModule()
dm.setup()


model = CLIPModel()



## Loggers
logger:pl_loggers.TensorBoardLogger = pl_loggers.TensorBoardLogger(save_dir='logs/',name= "lightning_logs",log_graph=True) 

## CallBacks
call_backs = [
    TQDMProgressBar(refresh_rate=10),
    ModelCheckpoint(
        monitor="val_loss", dirpath=os.path.join('logs','chkpoints'), filename="{epoch:02d}",save_top_k=1,
    ),
    DeviceStatsMonitor(cpu_stats=True),
    LearningRateMonitor(logging_interval='step')
]


trainer = pl.Trainer(precision=16, max_epochs=3, accelerator="gpu",logger=logger, profiler='pytorch',callbacks=call_backs,limit_train_batches=0.2,limit_test_batches=0.2,limit_val_batches=0.2)


trainer.fit(model,datamodule=dm)




_, valid_df = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")
find_matches(model,
             image_embeddings.to("cuda"),
             query="horse running",
             image_filenames=valid_df['image'].values,
             n=9)
find_matches(model,
             image_embeddings.to("cuda"),
             query="people dancing",
             image_filenames=valid_df['image'].values,
             n=9)



# from dataset import get_transforms
# test_transform = get_transforms(mode='test')
# dataset = CLIPDataset(
#         dm.train_df["image"].values,
#         dm.train_df["caption"].values,
#         tokenizer=dm.tokenizer,
#         transforms=test_transform,
#     )
# dl = torch.utils.data.DataLoader( dataset, batch_size= 1_000, shuffle=True )
# batch = next(iter(dl))
# print(batch['image'].device)
# torch.save(batch['image'], "10kcpu_imgs.pt")
# training_images =  torch.load('10kcpu_imgs.pt')
# training_images.to('cuda:0').shape
# features1 = model.cuda().image_encoder( training_images.cuda()[:50,:,:,:] )
# features2 = model.cuda().image_projection(features1)
# print(features2.shape)

# import torchvision
# batch['image'].shape
# random_image = batch['image'][324]
# img_features_1 = model.image_encoder( torchvision.transforms.Resize((64,64))( random_image ).unsqueeze(0).cuda() )
# img_features_2 = model.image_projection(img_features_1)
# print(random_image.shape, img_features_2.shape)
# with torch.no_grad():
#     cost = img_features_2 @ features2.T
# plt.imshow(random_image.permute(1,2,0).cpu())
# plt.show()

# print('*'*200)
# for i in cost.argsort(dim=-1,descending=True)[:,:5].flatten().cpu():
#     plt.imshow(
#         torchvision.transforms.Normalize(mean= (-.5,-.5,-.5), std=(1,1,1))(
#             training_images[i.item()]
#         ).permute(1,2,0)
#     )
#     plt.show()