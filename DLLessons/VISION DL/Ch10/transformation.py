from albumentations.pytorch import ToTensorV2 
import albumentations as A

test_transforms = A.Compose([
            A.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
                always_apply=True
            ),
            ToTensorV2()
        ])


train_transforms = A.Compose([
    A.Normalize(
        mean=(0.49139968, 0.48215841,0.44653091), 
        std=(0.24703223,0.24348513,0.26158784)
    ),
    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
    A.RandomCrop(height=32, width=32, always_apply=True),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(
               shift_limit=0.0625, 
               scale_limit=0.1, 
               rotate_limit=45, 
               interpolation=1, 
               border_mode=4, 
               p=0.5
            ),
    A.CoarseDropout(
        max_holes=1,
        max_height=8,
        max_width=8,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=[0.49139968, 0.48215841, 0.44653091],
        always_apply=True
        ),
        ToTensorV2(),
])