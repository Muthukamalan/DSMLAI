import os 
import torch

debug = True
data_dir = r"C:\Users\muthu\GitHub\DATA 📁\Flicker"
image_path    = os.path.join(data_dir,"Images")
captions_path = os.path.join(data_dir,'captions.txt')
batch_size = 8
num_workers = 0

# optimizer
lr = 1e-3
weight_decay = 1e-3
# scheduler
patience = 2
factor = 0.5
# trainer
epochs = 5

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Image-encoder
model_name = 'resnet50'
image_embedding = 2048
## Text-encoder
text_encoder_model = "distilbert-base-uncased"
text_tokenizer = "distilbert-base-uncased"
text_embedding = 768
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1