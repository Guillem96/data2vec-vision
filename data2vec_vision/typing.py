from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple, Union

import torch
from PIL import Image

PathLike = Union[Path, str]
ImageOrTensor = Union[torch.Tensor, Image.Image]
MaskingFn = Callable[[ImageOrTensor], Tuple[ImageOrTensor, torch.Tensor]]
TransformFn = Callable[[ImageOrTensor], ImageOrTensor]
DataPoint = Tuple[ImageOrTensor, Tuple[ImageOrTensor, torch.Tensor]]


@dataclass(frozen=True)
class Data2VecSample(object):
    image: ImageOrTensor
    masked_image: ImageOrTensor
    bool_mask: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(astuple(self))
