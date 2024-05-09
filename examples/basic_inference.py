import torch
from xops import xrearrange, xreduce

from time import perf_counter

device = torch.device('mps')

tensor1 = torch.ones(1, 1, 1, 1).to(device)

start = perf_counter()
tensor1a = xrearrange(tensor1, '...->...')
print(tensor1a.shape)
end = perf_counter()
print(f'xrearrange took {end - start:.6f} seconds')

# start = perf_counter()
# tensor1b = xreduce(tensor1, 'b h w c -> b c', 'mean')
# end = perf_counter()
# print(f'xreduce took {end - start:.6f} seconds')

# print(tensor1a.shape)
# print(tensor1b.shape)

# from einops import rearrange, reduce

# start = perf_counter()
# tensor2a = rearrange(tensor1, 'b c h w -> (b h) (c w)')
# end = perf_counter()
# print(f'rearrange took {end - start:.6f} seconds')

# # start = perf_counter()
# tensor2b = reduce(tensor1, 'b h w c -> b c', 'mean')
# end = perf_counter()
# print(f'reduce took {end - start:.6f} seconds')

# print(tensor2a.shape)
# print(tensor2b.shape)

# closeness1 = (torch.isclose(tensor1a, tensor2a).sum().item()) / tensor1a.numel()
# closeness2 = (torch.isclose(tensor1b, tensor2b).sum().item()) / tensor1b.numel()
# print(f'Tensors are {closeness1 * 100:.2f}% similar')
# print(f'Tensors are {closeness2 * 100:.2f}% similar')
