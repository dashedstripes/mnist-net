import torch

# I need to understand how tensor shapes work

x = torch.tensor(1)
# x = tensor(1)
# x = torch.Size([])

# x.size() and x.shape are equivalent, x.shape was added to more closely match numpy

x = torch.tensor([1])
# x = tensor([1])
# x.size() = torch.Size([1])

x = torch.tensor([1, 2])
# x = tensor([1, 2])
# x.size() = torch.Size([2])

x = torch.tensor([[1]])
# x = tensor([[1]])
# x.size() = torch.Size([1, 1])

# size is out to in, so starting at the top of [[1]], there is one dimension, with one dimension inside it.
# therefore, I'm going to guess the following to be [2, 1]

x = torch.tensor([[1], [1]])
# x = tensor([[1], [1]])
# x.size = torch.Size([2, 1])

# stepping it up a notch, what is this?
# x = torch.tensor([[1, 2], [1]])
# this is an invalid tensor, as it is not balanced

# we can fix this by balancing the tensor
x = torch.tensor([[1, 2], [3, 4]])
# x.size() = torch.Size([2, 2])

x = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])
# x.size() = torch.Size([4, 1, 2])

# x = torch.tensor([[[1, 2]], [[3, 4]], [[5]], [[7, 8]]])
# ValueError: expected sequence of length 2 at dim 2 (got 1)
# once we are at dimension 2, we're expecting 2 items, but we got 1
# the correct dimensions for this tensor is [4, 1, 2], so our issue happens in our 3rd dimension