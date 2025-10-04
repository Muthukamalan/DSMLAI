from torchvision import transforms 

train_transforms = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465),
    #     (0.2470, 0.2435, 0.2616)
    # ),
    # transforms.Resize((32,32),antialias=False),
    # transforms.RandomHorizontalFlip()
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),
    transforms.Resize((32,32),antialias=False),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation((-10., 10.), fill=1),
    
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
    transforms.Resize((32,32),antialias=False)
])