from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
shuffle_datasets=False

# https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
train_data = datasets.MNIST(root="data", download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_datasets)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_datasets)

train_features, train_labels = next(iter(test_dataloader))
# print(train_features, train_labels)

# render the first feature to the screen, so I can check which label it belongs to
plt.imshow(train_features[1].reshape(28,28), cmap="gray")
plt.show()
