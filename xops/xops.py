from typing import List, Union, Dict, Tuple, TypeVar, Any
from functools import reduce, lru_cache
from .parsing import ParsedExpression
from ._backends import get_backend
from itertools import islice

Tensor = TypeVar("Tensor")
Shape = Tuple[int, ...]

class XOpsError(Exception):
    pass

def flatten_grouped_expression(expr: Tuple[Tuple[Union[str]]]) -> Tuple[Union[str, Tuple[str]]]:
    return tuple(
        group[0] if len(group) == 1 else tuple(group)
        for group in expr
    )

def infer_shape(tensor_shape: Tuple[int, ...], expr: Tuple[Tuple[Union[str]]]) -> Dict[Union[str, Tuple[str]], Union[int, List[int]]]:
    flattened_expr = flatten_grouped_expression(expr)
    tensor_shape = list(tensor_shape)
    flattened_expr_len = len(flattened_expr)

    shape = {}
    idx = 0
    for i, symbol in enumerate(flattened_expr):
        remaining_len = flattened_expr_len - i - 1
        if symbol == '…':
            end_idx = len(tensor_shape) - remaining_len
            shape[symbol] = tensor_shape[idx:end_idx]
            idx = end_idx
        elif isinstance(symbol, tuple):
            product = 1
            for dim in islice(tensor_shape, idx, idx + len(symbol)):
                product *= dim
            shape[symbol] = product
            idx += len(symbol)
        else:
            shape[symbol] = tensor_shape[idx]
            idx += 1

    if idx != len(tensor_shape):
        raise ValueError("Shape mismatch between tensor and pattern.")

    return shape

def get_labels_to_indices(left_expr: Tuple[Tuple[Union[str]]], shape: Tuple[int, ...]) -> Dict[Union[str, Tuple[str]], List[int]]:
    """ Create a mapping from labels to indices based on left_expr and input tensor shape. """
    labels_to_indices = {}
    current_index = 0

    # Determine the number of dimensions required for `…`
    num_free_dims = len(shape) - sum(len(expr) for expr in left_expr if expr != ('…',))
    for expr in left_expr:
        if expr == ('…',):
            ellipsis_indices = list(range(current_index, current_index + num_free_dims))
            labels_to_indices['…'] = ellipsis_indices
            current_index += num_free_dims
        else:
            labels_to_indices[expr[0]] = current_index
            current_index += 1

    return labels_to_indices

def create_permutation_indices(shape: Tuple[int, ...], left_expr: Tuple[Tuple[Union[str]]], right_expr_flat: Tuple[Union[str, Tuple[str]]]) -> List[int]:
    labels_to_indices = get_labels_to_indices(left_expr, shape)

    permutation_indices = []
    for dim in right_expr_flat:
        if dim == '…':
            permutation_indices.extend(labels_to_indices['…'])
        elif isinstance(dim, tuple):
            for sub_dim in dim:
                permutation_indices.append(labels_to_indices[sub_dim])
        else:
            permutation_indices.append(labels_to_indices[dim])

    return permutation_indices

def xrearrange(tensor: Tensor, pattern: str) -> Tensor:
    tensor = get_backend(tensor)
    left, right = pattern.split('->')
    
    # Parsing and mapping expression only once
    left_parsed_expr = tuple(map(tuple, ParsedExpression(left).composition))
    right_parsed_expr = tuple(map(tuple, ParsedExpression(right).composition))

    left_expr = tuple(map(tuple, left_parsed_expr))
    right_expr = tuple(map(tuple, right_parsed_expr))

    # Flatten grouped expressions only once
    right_expr_flat = flatten_grouped_expression(right_expr)

    # Infer shape once for all left symbols
    left_shape = infer_shape(tuple(tensor.shape), left_expr)

    # Compute right shape using a preallocated dictionary
    right_shape = {}
    for symbol in right_expr_flat:
        if symbol == '…':
            right_shape[symbol] = left_shape['…']
        else:
            if isinstance(symbol, tuple):
                right_shape[symbol] = reduce((lambda x, y: x * y), map(lambda s: left_shape[s], symbol))
            else:
                right_shape[symbol] = left_shape[symbol]

    output_shape = [right_shape[symbol] for symbol in right_expr_flat]

    flattened_output_shape = [dim for symbol in output_shape for dim in (symbol if isinstance(symbol, list) else [symbol])]

    permutation = create_permutation_indices(tuple(tensor.shape), left_expr, right_expr_flat)
    return tensor.permute(permutation).reshape(*flattened_output_shape)

def xreduce(tensor: Tensor, pattern: str, reduction: str, **reduction_kwargs) -> Tensor:
    left, right = pattern.split('->')
    left_expr = tuple(map(tuple, ParsedExpression(left).composition))
    right_expr = tuple(map(tuple, ParsedExpression(right).composition))

    reduction_axes = []
    right_symbols = set(flatten_grouped_expression(right_expr))
    for i, symbol in enumerate(flatten_grouped_expression(left_expr)):
        if symbol not in right_symbols:
            reduction_axes.append(i)

    if reduction == 'sum':
        return tensor.sum(dim=reduction_axes, **reduction_kwargs)
    elif reduction == 'mean':
        return tensor.mean(dim=reduction_axes, **reduction_kwargs)
    elif reduction == 'max':
        for axis in reduction_axes:
            tensor, _ = tensor.max(dim=axis, **reduction_kwargs)
        return tensor
    elif reduction == 'min':
        for axis in reduction_axes:
            tensor, _ = tensor.min(dim=axis, **reduction_kwargs)
        return tensor
    else:
        raise XOpsError(f"Unsupported reduction operation: {reduction}")