import torch
from torch import nn

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

m = nn.Softmax(dim=1)

train_logits = torch.tensor([
  [ 1.1391e-02, -5.2017e-02,  3.2917e-01,  1.9043e-01, -4.0752e-02, 2.4881e-02, -4.2587e-02, -3.4064e-01,  1.5641e-01, -1.3051e-01],
  [ 1.3456e-01, -1.0080e-02,  1.4288e-01, -6.0055e-02, -4.3025e-02,-3.8400e-02,  3.6658e-01, -1.9184e-01, -2.0695e-01, -2.7804e-01],
        [ 5.8732e-02, -5.6161e-02,  6.9488e-02, -1.0273e-02,  1.1555e-01,
          4.1659e-02, -6.0482e-02,  3.1923e-02, -4.9068e-02,  4.5057e-02],
        [-1.2683e-01, -6.0807e-02,  6.6595e-03, -4.0041e-02,  5.2191e-02,
          1.4873e-02, -1.4097e-01,  3.3162e-02, -2.4361e-02,  9.3167e-02],
        [ 8.2705e-02, -3.3749e-01,  1.7743e-02, -1.1692e-01,  1.7018e-01,
         -1.7056e-02,  7.6525e-03, -8.9546e-02, -1.8201e-01, -8.8059e-02],
        [-1.4639e-01,  2.8039e-02, -9.6365e-02, -4.9721e-02,  5.3704e-02,
         -4.7066e-02, -2.3368e-01,  5.8484e-02,  3.9915e-02,  1.0663e-01],
        [ 4.1222e-01, -3.0290e-01,  4.5713e-01,  1.8150e-01, -1.1862e-01,
          1.0633e-02,  1.2905e-01, -3.8176e-01,  2.4396e-02, -3.7151e-01],
        [-1.2727e-02, -5.6711e-02, -4.6184e-02, -1.3647e-01,  6.1193e-02,
         -3.0120e-02, -1.0351e-01,  1.9573e-02, -2.8421e-02,  9.3729e-02],
        [ 2.6071e-01, -2.1229e-01,  2.0780e-01,  4.0209e-02, -2.2723e-02,
         -9.0139e-02,  2.7247e-01, -2.3000e-01, -1.5351e-01, -2.2054e-01],
        [ 6.8748e-01, -1.3492e-01,  2.6432e-01, -6.9069e-04, -1.7808e-01,
         -1.3416e-01, -6.8734e-03, -1.4737e-01, -2.1395e-02, -3.7919e-01],
        [-3.9969e-02, -2.8623e-02, -1.8623e-02, -2.6066e-03, -6.8820e-02,
         -2.7707e-02, -2.0250e-01,  1.6401e-01, -6.3155e-03,  1.7648e-01],
        [ 6.9880e-01, -1.0756e-01,  2.2222e-01, -8.0749e-02, -2.5455e-01,
         -2.0902e-01, -1.4153e-01, -1.1977e-01,  3.2159e-02, -3.0940e-01],
        [-2.6260e-01,  9.9670e-02, -5.6084e-02, -2.0678e-02,  7.5877e-02,
         -1.2086e-01,  1.2282e-02,  7.6947e-02,  4.1220e-02, -2.8071e-02],
        [-2.4441e-02,  2.9261e-01,  7.5300e-02,  1.0306e-02, -1.9816e-01,
         -1.7478e-01, -3.0713e-01, -1.5741e-01,  3.2532e-01,  5.9121e-03],
        [ 4.1169e-01, -2.1609e-01,  3.6442e-01,  8.0202e-02, -1.0037e-01,
         -5.8907e-02,  5.8946e-02, -3.0975e-01, -1.4347e-02, -3.1795e-01],
        [ 8.1136e-01, -1.2956e-01,  2.8038e-01, -1.2117e-01, -2.8956e-01,
         -1.8589e-01,  1.0022e-01, -1.2887e-01, -2.2079e-02, -3.2357e-01]])

test_logits = torch.tensor([
  [-3.7889e+00,  9.9500e-01,  2.2073e+00,  1.3584e+00, -2.9906e+00, 3.1233e-01, -1.2841e+00, -1.5940e+00,  6.0177e+00,  4.5874e-01],
  [-7.1989e-02, -1.1163e-01,  5.5536e+00,  5.7602e+00, -7.9822e+00, 1.7148e+00,  8.6180e-02,  7.4268e-01, -8.1101e-01, -3.8641e+00],
        [-1.1178e+00, -8.6086e-02,  5.9254e-02,  8.4579e+00, -2.4118e+00,
          3.7496e+00, -4.7370e+00, -2.9496e+00,  2.4791e+00, -1.6429e+00],
        [-1.0962e+00, -5.9508e+00, -2.2122e+00,  1.5089e+00,  2.6458e+00,
          1.0491e+00, -2.4550e+00,  1.5805e+00,  7.8723e-01,  7.0677e+00],
        [-2.5717e+00, -6.6146e+00, -5.1673e+00, -1.3795e+00,  1.0764e+01,
          2.7193e+00, -2.6818e-01,  2.2648e-01,  1.2320e+00,  6.0670e+00],
        [ 5.0644e+00, -4.5374e+00,  5.8452e-01,  1.4201e+00, -2.7772e+00,
          3.4789e+00, -1.4757e+00, -9.0056e-01,  1.9929e+00, -9.8105e-01],
        [-2.9258e+00,  2.2081e+00,  5.1541e-01,  2.4502e+00, -1.5042e+00,
          1.0674e+00, -1.5219e+00, -1.4777e+00,  1.9081e+00, -6.8912e-01],
        [-1.0650e+00, -5.3243e+00,  3.8424e-02,  2.2407e+00, -2.0729e+00,
         -1.7633e+00, -7.6233e+00,  1.1576e+01,  7.0792e-03,  5.9670e+00],
        [-3.0118e+00,  1.0524e+00,  1.2238e+00,  8.6744e+00, -3.4244e+00,
          1.9863e+00, -5.5480e+00, -2.1210e+00,  2.1861e+00,  1.0728e-01],
        [-3.3757e-01, -5.0017e-01,  9.8380e+00,  2.7458e+00, -4.1518e+00,
         -1.2942e+00,  6.8549e-01, -4.1247e+00,  2.7406e+00, -4.1251e+00],
        [-2.9271e-01, -1.6968e+00, -1.5235e+00,  5.9274e+00, -1.0948e+00,
          2.5640e+00, -1.7374e+00, -7.2092e-01, -1.4108e-01,  3.7288e-01],
        [ 5.1248e-01, -2.2278e+00, -8.4418e-01,  7.8897e+00, -3.3049e+00,
          4.8411e+00, -1.7728e+00, -2.0932e+00,  6.6175e-01, -1.5131e+00],
        [ 2.8116e-01, -5.6014e+00, -2.1067e+00,  2.1796e-02,  9.4515e-01,
          6.5860e-01, -3.3590e+00,  5.7187e+00, -3.3674e-01,  5.9479e+00],
        [ 3.9295e-02, -2.7947e+00, -1.3680e-01,  6.5017e+00, -4.6163e-01,
          4.0705e+00, -4.7023e+00, -2.4004e+00,  2.8947e+00, -4.6742e-01],
        [-3.6286e+00, -1.2936e+00, -3.5983e+00, -1.6749e-01,  5.0901e+00,
          1.6979e+00, -1.5317e+00,  1.0482e+00,  1.4511e+00,  3.1746e+00],
        [ 9.5590e+00, -1.2404e+01,  6.4544e-01,  3.0953e-01, -2.1467e+00,
          3.2304e+00, -2.4350e-01,  1.0052e+00,  1.8461e+00,  2.8443e+00]])

print(train_logits.shape)
print(test_logits.shape)

print(m(train_logits))
print(m(test_logits))