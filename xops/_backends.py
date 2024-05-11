from typing import Any, TypeVar

Tensor = TypeVar("Tensor")

class AbstractBackend:
    """Base backend class for xops."""

    framework_name: str

    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    @property
    def is_appropriate_type(self) -> bool:
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        raise NotImplementedError
    
    @property
    def dtype(self) -> str:
        raise NotImplementedError
    
    @property
    def to_numpy(self) -> Tensor:
        raise NotImplementedError
    
    def reshape(self, shape: tuple) -> Tensor:
        raise NotImplementedError
    
    def permute(self, dims: tuple) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        return "<xops backend for {}>".format(self.framework_name)
    
class PyTorchBackend(AbstractBackend):
    """PyTorch backend class for xops."""

    framework_name = "PyTorch"

    def __init__(self, tensor: Tensor):
        import torch
        self.torch = torch

        self.tensor = tensor

    @property
    def is_appropriate_type(self) -> bool:
        return isinstance(self.tensor, self.torch.Tensor)

    @property
    def shape(self) -> tuple:
        return self.tensor.shape
    
    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)
    
    @property
    def to_numpy(self) -> Tensor:
        return self.tensor.numpy()
    
    def reshape(self, shape: tuple) -> Tensor:
        return self.tensor.reshape(shape)
    
    def permute(self, dims: tuple) -> Tensor:
        return self.tensor.permute(dims)

    def __repr__(self):
        return "<xops backend for {}>".format(self.framework_name)
    
class TensorFlowBackend(AbstractBackend):
    """TensorFlow backend class for xops."""

    framework_name = "TensorFlow"

    def __init__(self, tensor: Tensor):
        import tensorflow as tf
        self.tf = tf

        self.tensor = tensor

    @property
    def is_appropriate_type(self) -> bool:
        return isinstance(self.tensor, self.tf.Tensor)

    @property
    def shape(self) -> tuple:
        return self.tensor.shape
    
    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)
    
    @property
    def to_numpy(self) -> Tensor:
        return self.tensor.numpy()
    
    def reshape(self, shape: tuple) -> Tensor:
        return self.tf.reshape(self.tensor, shape)

    def permute(self, dims: tuple) -> Tensor:
        return self.tf.transpose(self.tensor, dims)

    def __repr__(self):
        return "<xops backend for {}>".format(self.framework_name)
    
class JAXBackend(AbstractBackend):
    """JAX backend class for xops."""

    framework_name = "JAX"

    def __init__(self, tensor: Tensor):
        import jax.numpy as jnp
        self.jnp = jnp

        self.tensor = tensor

    @property
    def is_appropriate_type(self) -> bool:
        return isinstance(self.tensor, self.jnp.ndarray)

    @property
    def shape(self) -> tuple:
        return self.tensor.shape
    
    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)
    
    @property
    def to_numpy(self) -> Tensor:
        return self.tensor.copy()
    
    def reshape(self, shape: tuple) -> Tensor:
        return self.jnp.reshape(self.tensor, shape)
    
    def permute(self, dims: tuple) -> Tensor:
        return self.jnp.transpose(self.tensor, dims)

    def __repr__(self):
        return "<xops backend for {}>".format(self.framework_name)
    
class NumPyBackend(AbstractBackend):
    """NumPy backend class for xops."""

    framework_name = "NumPy"

    def __init__(self, tensor: Tensor):
        import numpy as np
        self.np = np

        self.tensor = tensor

    @property
    def is_appropriate_type(self) -> bool:
        return isinstance(self.tensor, self.np.ndarray)

    @property
    def shape(self) -> tuple:
        return self.tensor.shape
    
    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)
    
    @property
    def to_numpy(self) -> Tensor:
        return self.tensor.copy()
    
    def reshape(self, shape: tuple) -> Tensor:
        return self.tensor.reshape(shape)
    
    def permute(self, dims: tuple) -> Tensor:
        return self.tensor.transpose(dims)

    def __repr__(self):
        return "<xops backend for {}>".format(self.framework_name)

def get_backend(tensor) -> AbstractBackend:
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    backends = AbstractBackend.__subclasses__()
    for backend in backends:
        if backend(tensor).is_appropriate_type:
            return backend(tensor)
    raise RuntimeError("Tensor type unknown to xops {}".format(type(tensor)))

if __name__ == '__main__':
    import torch
    import tensorflow as tf
    import jax.numpy as jnp
    import numpy as np

    tensor1 = torch.ones((2, 3, 4))
    tensor2 = tf.ones((2, 3, 4))
    tensor3 = jnp.ones((2, 3, 4))
    tensor4 = np.ones((2, 3, 4))

    print(get_backend(tensor1))
    print(get_backend(tensor2))
    print(get_backend(tensor3))
    print(get_backend(tensor4))