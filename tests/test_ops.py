from xops import xrearrange, xreduce
from einops import rearrange, reduce
import torch
from time import perf_counter

identity_patterns = [
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
]

equivalent_rearrange_patterns = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]

equivalent_reduction_patterns = [
    ("a b c d e -> ", " ... ->  "),
    ("a b c d e -> (e a)", "a ... e -> (e a)"),
    ("a b c d e -> d (a e)", " a b c d e ... -> d (a e) "),
    ("a b c d e -> (a b)", " ... c d e  -> (...) "),
]

def test_identity_patterns():
    for pattern in identity_patterns:
        tensor = torch.zeros(1, 1, 1, 1, 1)
        tensor1 = xrearrange(tensor, pattern)
        tensor2 = rearrange(tensor, pattern)
        assert torch.isclose(tensor1, tensor2).all()

def test_equivalent_rearrange_patterns():
    for pattern1, pattern2 in equivalent_rearrange_patterns:
        tensor = torch.zeros(1, 1, 1, 1, 1)
        tensor1 = xrearrange(tensor, pattern1)
        tensor2 = rearrange(tensor, pattern2)
        assert torch.isclose(tensor1, tensor2).all()

def test_equivalent_reduction_patterns():
    for pattern1, pattern2 in equivalent_reduction_patterns:
        tensor = torch.zeros(1, 1, 1, 1, 1)
        tensor1 = xreduce(tensor, pattern1, 'mean')
        tensor2 = reduce(tensor, pattern2, 'mean')
        assert torch.isclose(tensor1, tensor2).all()

def test_xrearrange_speed():
    tensor = torch.rand(32, 128, 1024, 1)
    start = perf_counter()
    tensor1 = xrearrange(tensor, 'b c h w -> (b h) (c w)')
    end = perf_counter()
    time_xops = end - start

    start = perf_counter()
    tensor2 = rearrange(tensor, 'b c h w -> (b h) (c w)')
    end = perf_counter()
    time_einops = end - start

    assert torch.isclose(tensor1, tensor2).all()

    print(f'xrearrange took {time_xops:.6f} seconds')
    print(f'rearrange took {time_einops:.6f} seconds')
    print(f'Speed increase: {time_einops / time_xops:.2f}x')

def test_xreduce_speed():
    tensor = torch.rand(32, 128, 1024, 1)
    start = perf_counter()
    tensor1 = xreduce(tensor, 'b h w c -> b c', 'mean')
    end = perf_counter()
    time_xops = end - start

    start = perf_counter()
    tensor2 = reduce(tensor, 'b h w c -> b c', 'mean')
    end = perf_counter()
    time_einops = end - start

    assert torch.isclose(tensor1, tensor2).all()

    print(f'xreduce took {time_xops:.6f} seconds')
    print(f'reduce took {time_einops:.6f} seconds')
    print(f'Speed increase: {time_einops / time_xops:.2f}x')

if __name__ == '__main__':
    test_identity_patterns()
    test_equivalent_rearrange_patterns()
    test_equivalent_reduction_patterns()
    test_xrearrange_speed()
    test_xreduce_speed()