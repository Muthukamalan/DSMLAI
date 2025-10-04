import torch
class CFG:
    debug = False
    image_path = "/home/muthu/GitHub/DATA 📁/flickr30k_images/flickr30k_images" #"/content/flickr30k_images/flickr30k_images"
    captions_path = "/home/muthu/GitHub/DATA 📁/flickr30k_images/"
    batch_size = 8
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet18' # 'resnet50'
    image_embedding = 512
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = False # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1