import torch
from torchvision import transforms


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


IMG_RESOLUTION:int = 128 #256

img_trasfroms:transforms.Compose = transforms.Compose([
    transforms.Resize( (IMG_RESOLUTION, IMG_RESOLUTION ),interpolation=transforms.InterpolationMode.NEAREST_EXACT ),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean,std=std)
])





mask_transforms = transforms.Compose(
    [
        transforms.PILToTensor(),                                        
        transforms.Resize( (IMG_RESOLUTION, IMG_RESOLUTION),interpolation=transforms.InterpolationMode.NEAREST_EXACT),        
        
        # ONE HOT  of (C,H,W)  >> [  (C[0],H,W)  ,  (C[1],H,W), (C[2],H,W)  ]
        transforms.Lambda(lambda x: (x - 1).squeeze().type(torch.LongTensor) )
    ]
)