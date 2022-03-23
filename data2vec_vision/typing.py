from pathlib import Path
from typing import Callable, NamedTuple, Tuple, Union

import torch
from PIL import Image

PathLike = Union[Path, str]
ImageOrTensor = Union[torch.Tensor, Image.Image]
MaskingFn = Callable[[ImageOrTensor], Tuple[ImageOrTensor, torch.Tensor]]
TransformFn = Callable[[ImageOrTensor], ImageOrTensor]
DataPoint = Tuple[ImageOrTensor, Tuple[ImageOrTensor, torch.Tensor]]


class Data2VecSample(NamedTuple):
    image: ImageOrTensor
    masked_image: ImageOrTensor
    bool_mask: torch.Tensor
