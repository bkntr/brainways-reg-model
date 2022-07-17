import random
from copy import deepcopy
from typing import Tuple, Union, Optional

import kornia as K
import torch

from duracell.slice_generator.slice_generator_sample import SliceGeneratorSample


class RandomAffine:
    """
    Random scale input image 
    """

    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float]] = 0,
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[
            Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]
        ] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float]]] = None,
        scale_y: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
    ):
        """
        
        :param degrees: 
        :param translate: 
        :param scale: 
        :param shear: 
        :param scale_y: additional random y scaling, on top of the scale parameter
        """
        self.random_affine = K.augmentation.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            p=1.0,
            same_on_batch=True,
        )
        self.random_affine_nearest = K.augmentation.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            p=1.0,
            same_on_batch=True,
            resample="nearest",
        )
        self.scale_y = scale_y

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        params = self.random_affine.generate_parameters(
            torch.Size((1,) + sample.image.shape)
        )

        if self.scale_y is not None:
            params["scale"][:, 1] *= random.uniform(self.scale_y[0], self.scale_y[1])

        # scaling is not centered correctly in Kornia, so we do the centering ourselves
        params_zero_center = deepcopy(params)
        torch.zero_(params_zero_center["center"])

        transform = self.random_affine.compute_transformation(
            sample.image, params_zero_center
        )
        zero_center_matrix = torch.tensor(
            [
                [1, 0, -params["center"][0, 0]],
                [0, 1, -params["center"][0, 1]],
                [0, 0, 1],
            ],
            device=transform.device,
            dtype=transform.dtype,
        )
        center_matrix = torch.tensor(
            [
                [1, 0, params["center"][0, 0]],
                [0, 1, params["center"][0, 1]],
                [0, 0, 1],
            ],  # black
            device=transform.device,
            dtype=transform.dtype,
        )
        transform = center_matrix @ transform @ zero_center_matrix
        sample.image = self.random_affine.apply_transform(
            sample.image, params, transform
        )
        sample.regions = self.random_affine_nearest.apply_transform(
            sample.regions, params, transform
        )
        sample.hemispheres = self.random_affine_nearest.apply_transform(
            sample.hemispheres, params, transform
        )
        return sample
