from typing import Optional, Sequence

import torch
from PIL import Image

from data2vec_vision.typing import (Data2VecSample, MaskingFn, PathLike,
                                    TransformFn)


class Data2VecDataset(torch.utils.data.Dataset):
    """Indexable dataset that generates samples for Data2Vec vision model.

    Parameters
    ----------
    paths : Sequence[PathLike]
        List of image paths.
    masking_fn : MaskingFn
        Function that masks a given image (after applying the transforms) 
        and returns the masked image as well as the applied boolean mask.
    before_masking_tfms : Optional[TransformFn]
        Transformations that will be applied to the image before masking it.
    after_masking_tfms : Optional[TransformFn]
        Transformations that will be applied separately to the image and the
        masked image. Usually used for formatting the patches to sequence with
        `transforms.PatchesToSequence`.
    """

    def __init__(self,
                 paths: Sequence[PathLike],
                 masking_fn: MaskingFn,
                 before_masking_tfms: Optional[TransformFn] = None,
                 after_masking_tfms: Optional[TransformFn] = None) -> None:
        self.paths = paths
        self.before_masking_tfms = before_masking_tfms
        self.after_masking_tfms = after_masking_tfms
        self.masking_fn = masking_fn

    def __getitem__(self, idx: int) -> Data2VecSample:
        im = Image.open(self.paths[idx]).convert('RGB')

        if self.before_masking_tfms is not None:
            im = self.before_masking_tfms(im)

        masked_im, bool_mask = self.masking_fn(im)

        if self.after_masking_tfms is not None:
            masked_im = self.after_masking_tfms(masked_im)
            im = self.after_masking_tfms(im)

        return Data2VecSample(image=im,
                              masked_image=masked_im,
                              bool_mask=bool_mask)

    def __len__(self) -> int:
        return len(self.paths)
