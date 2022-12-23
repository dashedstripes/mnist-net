import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

num_of_epochs = 20
learning_rate = 1e-2
batch_size = 64
shuffle_datasets=False


# https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
train_data = datasets.MNIST(root="data", download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_datasets)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_datasets)

train_features, train_labels = next(iter(train_dataloader))
print(train_features.shape)
print(train_labels.shape)
print(train_features.reshape(64, 28*28))
# train_features is pixels, train_labels is a single number 0-9, that is the same as the image
# print(nn.functional.one_hot(train_labels))

# render the first feature to the screen, so I can check which label it belongs to
# plt.imshow(train_features[6].reshape(28,28), cmap="gray")
# plt.show()

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

model = NeuralNetwork().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
m = nn.Softmax(dim=1)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    logits = model(X)
    # pred = m(logits)

    # https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814
    loss = loss_fn(logits ,y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # I am leaving this here because it's useful to know that the weights are updating, it helped me a lot during debugging
    # for name, param in model.named_parameters():
    #   print(name, param.grad)

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      logits = model(X)
      pred = m(logits)

      test_loss += loss_fn(pred, y).item()

      index_of_correct = torch.argmax(pred, dim=1)
      correct += torch.sum(index_of_correct == torch.argmax(y, dim=1)).item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run(save_model=False):
  epochs = num_of_epochs
  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
  print("Done!")

  if save_model:
    print("Saving Model")
    torch.save(model, 'models/rgb_nn.pth')
    print('Saved!')

run(save_model=False)