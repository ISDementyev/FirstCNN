import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
torch.manual_seed(0)

# valuable stats for normalization
mean_cifar = [0.4914, 0.4822, 0.4465]
epsilon = 0.0001 # generally set to 0.0001 if std == 0
std_cifar = [0.2470, 0.2435, 0.2616]

# Probability for randomly transforming the input photos
transform_p = 0.5
composed_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=transform_p),
                                         transforms.RandomVerticalFlip(p=transform_p),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_cifar, std_cifar)])

# Loading the dataset, and transforming the data
dset_test = CIFAR10(root="./data", train=True, transform=composed_transform, download=True)
dset_train = CIFAR10(root="./data", train=False, transform=composed_transform, download=False)

# Loading the data into DataLoader to be used for training function
bs = 200
train_loader = DataLoader(dataset=dset_train, batch_size=bs)
test_loader = DataLoader(dataset=dset_test, batch_size=bs)

# Create GPU object to load model into device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### The Model ###
class CNN(nn.Module):
    def __init__(self, in_channels=3, out_1=32, out_2=32, ks=3, img_size=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_1,
                              kernel_size=ks,
                              padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=out_1,
                              out_channels=out_2,
                              kernel_size=ks,
                              padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # CIFAR-10 inputs are 32x32 each, became 8x8 after 2 rounds of MaxPool2d
        dim_reduction = 8
        self.fc1 = nn.Linear(in_features=out_2 * dim_reduction * dim_reduction, out_features=10)
        self.bn_fc1 = nn.BatchNorm1d(10)


    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x

# Instantiate the model and load it onto the GPU (or CPU if a GPU is unavailable)
model = CNN(out_1=32, out_2=32)
model.to(device)

# Use Adam optimizer
wd=1e-3
lr=1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# Criterion for determining loss
criterion = nn.CrossEntropyLoss()

# training the model
model.train()

# for accuracy plotting
N_test = len(dset_test)

def train(model, criterion, optimizer, epochs=200):
    performance = {"loss": [], "validation accuracy": []}

    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for x, y in train_loader:
            # model.train()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            performance['loss'].append(loss.item())

        correct = 0
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            z_test = model(x_test)
            _, yhat = torch.max(z_test, 1)
            correct += (yhat == y_test).sum().item()

        accuracy = correct / N_test #* 100
        performance['validation accuracy'].append(accuracy)

    return performance

n_epochs = 100
model_metrics = train(model, criterion, optimizer, epochs=n_epochs)
plt.plot(model_metrics['validation accuracy'], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(model_metrics["loss"], label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
