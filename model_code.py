# from google.colab import drive
# drive.mount('/content/gdrive')

from torchsummary import summary
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import alexnet
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader , random_split


class ALEXNET(nn.Module):
    def __init__(self):
        super(ALEXNET, self).__init__()

        self.alexnet_model = alexnet(pretrained=True)

        self.features_conv = self.alexnet_model.features[:12]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.classifier = self.alexnet_model.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images here i used imagenet stats
])

def load_images_and_labels(folder_path):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):

            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.convert("RGB")

            image = image.resize((224,224))
            image_array = np.array(image) / 255.0
            # images.append(image_array)

            if ("target" in filename) & ("nontarget" not in filename):
                images.append(image_array)
                labels.append(1)
            elif "nontarget" in filename:
                images.append(image_array)
                labels.append(0)
            # else:
                # print(filename)
                # raise ValueError("Invalid filename format")

    return np.array(images), np.array(labels)



 # initialize the alexnet model
alexnet_model = ALEXNET()


for param in alexnet_model.alexnet_model.parameters():
    param.requires_grad = False


alexnet_model.alexnet_model.classifier[6] = nn.Linear(4096, 2)


alexnet_model.load_state_dict(torch.load('/content/gdrive/MyDrive/Colab Notebooks/FYP_coding/Finalizing_stage_01_models/alexnet_weights_of_scalogram_s5_256.pth'))

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet_model.alexnet_model.parameters())


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder('/content/gdrive/MyDrive/Colab Notebooks/FYP_coding/DataSet/xAI/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

# set the evaluation mode
alexnet_model.eval()

# get the image from the dataloader
img, _ = next(iter(dataloader))

img.requires_grad_(True)


# get the most likely prediction of the model
pred_class = alexnet_model(img).argmax(dim=1).numpy()[0]
pred = alexnet_model(img)

print("*****------Predicted Class--------*****: ", pred_class)