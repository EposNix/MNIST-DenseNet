import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# DenseNet block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = DenseBlock(in_channels=1, growth_rate=12, n_layers=4)
        self.block2 = DenseBlock(in_channels=1 + 12 * 4, growth_rate=12, n_layers=4)
        self.fc = nn.Linear((1 + 12 * 4 * 2) * 7 * 7, 10)

    def forward(self, x):
        out = self.block1(x)
        out = F.max_pool2d(out, 2)
        out = self.block2(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Print the total number of trainable parameters
total_params = count_parameters(model)
print(f'Total trainable parameters: {total_params}')

# Training function
def train(model, device, train_loader, optimizer):
    model.train()
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    return accuracy

# Training loop
epoch = 0
accuracy = 0

while accuracy < 100.0:
    epoch += 1
    accuracy = train(model, device, train_loader, optimizer)
    print(f'Train Epoch: {epoch} \tAccuracy: {accuracy:.2f}%')

print('Training complete. Model reached 100% accuracy.')

# Save the model weights
torch.save(model.state_dict(), 'mnist_densenet_weights.pth')

# Load and print the model weights
model_weights = torch.load('mnist_densenet_weights.pth')
for layer_name, weights in model_weights.items():
    print(f"Layer: {layer_name}\nWeights: {weights}\n")
