#Importing Modules
from torch import nn, save, load
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import ToTensor

# Ask user to pick dataset
SET_CHOICE = int(input("Please choose from the following datasets using the associated number:\n"
                       "1 - CIFAR-10\n2 - CIFAR-100\n3 - EMNIST\n4 - F-MNIST\n5 - MNIST\nYour Choice: "))

# Define dataset-specific class labels
classes_set = [
    None,  # Index 0 is unused

    # 1 - CIFAR-10
    ['airplane', 'automobile', 'bird', 'cat', 'deer',
     'dog', 'frog', 'horse', 'ship', 'truck'],

    # 2 - CIFAR-100
    ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
     'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
     'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
     'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
     'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
     'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
     'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
     'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
     'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
     'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],

    # 3 - EMNIST (Balanced)
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
     'f', 'g', 'h', 'n', 'q', 'r', 't'],

    # 4 - Fashion-MNIST
    ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],

    # 5 - MNIST
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
]

# Define root paths for where each dataset will be stored or loaded
root_set = [
    None,
    '/content/cifar-10-batches-py',
    '/content/cifar-100-python',
    '/content/EMNIST/processed',
    '/content/FashionMNIST/processed',
    '/content/MNIST/processed'
]

# Define dataset class map to make selection cleaner
dataset_map = {
    1: datasets.CIFAR10,
    2: datasets.CIFAR100,
    3: lambda **kwargs: datasets.EMNIST(split='balanced', **kwargs),
    4: datasets.FashionMNIST,
    5: datasets.MNIST
}

# Load the selected dataset using the dictionary
train = dataset_map[SET_CHOICE](root=root_set[SET_CHOICE], download=True, train=True, transform=ToTensor())

# Wrap the dataset in a DataLoader for batching and shuffling
data = DataLoader(train)

# Set the classes for future reference or display
classes = classes_set[SET_CHOICE]

# Define a basic CNN model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # First conv layer
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  # Second conv layer
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),  # Third conv layer
            nn.ReLU(),
            nn.Flatten(),  # Flatten output to feed into Linear
            nn.Linear(64 * 22 * 22, 10),  # Fully connected layer
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, optimizer, and loss function
modl = NeuralNet().to('cuda')
optimizer = Adam(modl.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training
EPOCHS = 5  # Number of times to go through the full dataset

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in data:
        X, y = batch
        X, y = X.to('cuda'), y.to('cuda')  # Move data to GPU

        optimizer.zero_grad()              # Reset gradients
        y_pred = modl(X)                   # Forward pass
        loss = loss_fn(y_pred, y)          # Compute loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Total Loss: {total_loss:.4f}")



