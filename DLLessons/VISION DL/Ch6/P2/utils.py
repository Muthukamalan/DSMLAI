from init import *

# Load Config
config = toml.load('config.toml')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(config['data']['dir_path'], train=True, download=False, transform=ToTensor())
test_data = datasets.MNIST(config['data']['dir_path'], train=False, download=False, transform=ToTensor())

def show__random_img():
    randnum = np.random.choice(train_data.targets.shape.numel())
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(train_data.data[randnum].squeeze(0), cmap='gray')
    plt.title(train_data.targets[randnum].item())
    plt.show()



def calculate_mean_std_mnist(datasets):
    data_loader = DataLoader(datasets,batch_size=1,shuffle=False)
    mean = torch.zeros(1);
    std = torch.zeros(1)
    num_samples = 0
    transform = transforms.ToTensor()
    for img in data_loader:
        image = img[0]
        image = image.squeeze()
        mean += image.mean()             # mean across channel sum for all pics
        std  += image.std()
        num_samples += 1

    mean /= num_samples
    std /= num_samples
    return (mean.item(),std.item())
