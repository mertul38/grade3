# --- imports ---
import skimage
import torch
import os
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import shutil
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import ImageReadMode, read_image
from PIL import Image, ImageOps
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from utils import *

# -- config --
TRAIN = False
VAL = False
TEST = True
DEVICE_ID = 'cpu'  # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
DATA_ROOT = 'ceng483-hw3-dataset'
torch.multiprocessing.set_start_method('spawn', force=True)

# ---- hyper-parameters ----
com_batch_size = 16
com_epochs_number_max = 5
com_learning_rate = 0.095
com_kernel_size = 3

class HW3ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root, device):
        super(HW3ImageFolder, self).__init__(root, transform=None)
        self.device = device

    def prepimg(self, img):
        return (transforms.functional.to_tensor(img) - 0.5) * 2  # normalize tensorized image from [0,1] to [-1,+1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.

        ############################################################################################################
        In order to obtain the path of the image with index index you may use following piece of code. As dataset goes
        over different indices you will collect image paths.

        myfile = open('test_images.txt', 'a')
        path = self.imgs[index][0]
        myfile.write(path)
        myfile.write('\n')
        myfile.close()
        ############################################################################################################
        """
        color_image, _ = super(HW3ImageFolder, self).__getitem__(index)  # Image object (PIL)
        print(f"color_image type: {type(color_image)}")
        grayscale_image = torchvision.transforms.functional.to_grayscale(color_image)
        return self.prepimg(grayscale_image).to(self.device), self.prepimg(color_image).to(self.device)

def get_loaders(batch_size,device):
    data_root = 'ceng483-hw3-dataset'
    train_set = HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = HW3ImageFolder(root='test', device=device)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader, test_loader

def save_rgb_img(img_arr, save_path):
    print(f"saved img: {img_arr.shape} to {save_path}: ex: {img_arr[0][0][0]}")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.imsave(save_path, img_arr)

def save_npy(estimations, filename):
    np.save(filename, estimations)

def save_estimation_names(estimation_names, filename):
    with open(filename, 'w') as f:
        for name in estimation_names:
            f.write(name)
            f.write('\n')

def showimg(img, gray=False):
    print("showimg")
    print(type(img))
    print(img.shape)
    plt.imshow(img, cmap="gray" if gray else None)
    plt.show()

# ---- ConvNet -----
class SuperResolutionNet(nn.Module):
    def __init__(self, kernel_size):
        super(SuperResolutionNet, self).__init__()
        self.layer_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        )
        self.layer_2 = nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        )
        self.layer_3 = nn.Conv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        )
        self.layer_4 = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)



    def forward(self, grayscale_image):
        x = self.forward_batch(grayscale_image)
        return x

    def forward_batch(self,grayscale_image):
        x = self.layer_1(grayscale_image)
        nn.BatchNorm2d(16)
        x = self.relu(x)
        x = self.layer_2(x)
        nn.BatchNorm2d(3)
        print("xxxxx")
        print(x)
        return x

    def forward_tanh(self,grayscale_image):
        x = self.layer_1(grayscale_image)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.tanh(x)
        print("xxxxx")
        print(x)
        return x

    def forward_16channel(self,grayscale_image):
        x = self.layer_1(grayscale_image)
        x = self.relu(x)
        x = self.layer_2(x)

        print("xxxxx")
        print(x)
device = torch.device(DEVICE_ID)
net = SuperResolutionNet(com_kernel_size).to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=com_learning_rate)

train_loader, val_loader, test_loader = get_loaders(com_batch_size,device)

# ---- training code -----

def train():
    ep_loss = []
    ep =[]
    print('training begins...')
    for epoch in range(com_epochs_number_max):
        ep.append(epoch)
        print(f"epoch {epoch + 1} of {com_epochs_number_max}")
        running_loss = 0.0  # training loss of the network
        for iteri, train_data in enumerate(train_loader, 0):
            print(f"    batch {iteri + 1} of {len(train_loader)}")

            inputs, targets = train_data  # inputs: low-resolution test, targets: high-resolution test.
            optimizer.zero_grad()  # zero the parameter gradients
            # do forward, backward, SGD step
            print(f"    gray - {inputs.shape}")
            print(f"    colored - {targets.shape}")
            preds = net(inputs)
            print(f"    pred - {preds.shape}")

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            # print loss
            running_loss += loss.item()

        print(f'Saving the model, end of epoch {epoch + 1} -'
              f' running loss: {running_loss}')
        ep_loss.append(running_loss)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))

    plt.clf()
    plt.plot(ep, ep_loss)
    plt.savefig(os.path.join(LOG_DIR, 'loss.png'))
    plt.show()

def val():
    checkpoint_path = os.path.join(LOG_DIR, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)
    print("CREATING VALIDATION")
    print("--*--*--*--*--*--*--*--*--")

    estimations = []
    ground_truths = []

    for iteri, data in enumerate(val_loader, 0):
        print(f"iteri {iteri} - data: {type(data)}")

        inputs, targets = data
        print(f"inputs - {inputs.shape} - {inputs.dtype}")
        print(f"targets - {targets.shape}")

        preds = net(inputs)
        print(f"preds - {preds.shape}")
        estimations.append(preds)
        ground_truths.append(targets)
        if iteri == 0:
            visualize_batch(inputs[:5], preds[:5], targets[:5], os.path.join(LOG_DIR, 'test_example.png'))
    print(f"estimations - {len(estimations)}")


    estimations = torch.cat(estimations, dim=0)
    estimations = estimations.detach().numpy()
    estimations = np.transpose(estimations, (0, 2, 3, 1))
    estimations = ((estimations + 1)/2)*255
    estimations = estimations.astype(np.uint8)

    ground_truths = torch.cat(ground_truths, dim=0)
    ground_truths = ground_truths.detach().numpy()
    ground_truths = np.transpose(ground_truths, (0, 2, 3, 1))
    ground_truths = ((ground_truths + 1)/2)*255
    ground_truths = ground_truths.astype(np.uint8)



    if os.path.exists("val_predictions"):
        shutil.rmtree("val_predictions")
    os.makedirs("val_predictions")
    save_npy(estimations, "../estimations.npy")
    save_npy(ground_truths, "val_predictions/ground_truths.npy")
    print(f"estimations - {estimations.shape}")

def test():
    checkpoint_path = os.path.join(LOG_DIR, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)
    print("CREATING VALIDATION")
    print("--*--*--*--*--*--*--*--*--")

    estimations = []
    ground_truths = []

    test_image_names = []
    test_images = []
    for x in test_loader:
        print(x)
        print("xxxx")
    for iteri in range(100):
        image_name = f"test/images/{iteri}.jpg"
        test_image_names.append(image_name)
        gray_image = ImageOps.grayscale(Image.open(image_name))
        gray_image = torchvision.transforms.functional.to_tensor(gray_image)
        print("gray_image - ", type(gray_image))
        print("gray_image - ", gray_image)
        gray_image = (gray_image - 0.5) * 2
        test_images.append(gray_image)

    print(f"test_images - {test_images[0]})")
    test_torch = torch.stack(test_images, dim=0)
    print(f"test_images - {len(test_images)}")
    print(f"test_torch - {test_torch.shape}")
    test_est = net(test_torch)
    print(test_est.shape)

    if os.path.exists("test_predictions"):
        shutil.rmtree("test_predictions")
    os.makedirs("test_predictions")
    save_npy(test_torch, "test_predictions/estimations_test.npy")
    save_estimation_names(test_image_names, "test_predictions/test_images.txt")
    visualize_batch(test_torch[:5], test_est[:5], test_est[:5], os.path.join(LOG_DIR, 'test_example.png'))


if TRAIN:
    train()
if VAL:
    val()
if TEST:
    test()


