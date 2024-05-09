__author__ = "Noah Kay"
__version__ = "0.0.0"


class XOpsError(RuntimeError):
    pass


__all__ = ["xrearrange", "xreduce"]

from .xops import xrearrange, xreduce