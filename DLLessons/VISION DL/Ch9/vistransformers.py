from torchvision import transforms 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# train_transforms = transforms.Compose([
#     # transforms.ToTensor(),
#     # transforms.Normalize(
#     #     (0.4914, 0.4822, 0.4465),
#     #     (0.2470, 0.2435, 0.2616)
#     # ),
#     # transforms.Resize((32,32),antialias=False),
#     # transforms.RandomHorizontalFlip()
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),
#     transforms.Resize((32,32),antialias=False),
#     transforms.ColorJitter(),
#     transforms.RandomHorizontalFlip(p=0.3),
#     transforms.RandomRotation((-10., 10.), fill=1),
    
# ])

# test_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         (0.4914, 0.4822, 0.4465),
#         (0.2470, 0.2435, 0.2616)
#     ),
#     transforms.Resize((32,32),antialias=False)
# ])

test_transforms = A.Compose([
            A.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
                always_apply=True
            ),
            ToTensorV2()
        ])


# cutout_transforms =  A.Compose([
#                                 A.PadIfNeeded(min_height=36, 
#                                             min_width=36, 
#                                             border_mode=cv2.BORDER_CONSTANT,
#                                             value=((0.4914, 0.4822, 0.4465))),
#                                 A.RandomCrop(32, 32),
#                                 A.Cutout(num_holes=4,max_h_size=8, max_w_size=4,fill_value=(0.4914, 0.4822, 0.4465)),
#                                 A.Normalize(mean=((0.4914, 0.4822, 0.4465)), 
#                                             std=(0.2470, 0.2435, 0.2616)),
#                                 ToTensorV2(),
#     ])

train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
               shift_limit=0.0625, 
               scale_limit=0.1, 
               rotate_limit=45, 
               interpolation=1, 
               border_mode=4, 
               p=0.5
            ),
            A.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
                always_apply=True
            ),
            A.CoarseDropout(
                max_holes=2, 
                max_height=8, 
                max_width=8, 
                fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=(0.4914, 0.4822, 0.4465),
                p=0.9
            ),
            A.RandomBrightnessContrast(p=0.2),
            # A.ToGray(p=0.1),
            ToTensorV2()
        ])