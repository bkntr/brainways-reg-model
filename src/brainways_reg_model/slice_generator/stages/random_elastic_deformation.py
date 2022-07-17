import random
from typing import Tuple

import kornia as K
import torch

from duracell.slice_generator.slice_generator_sample import SliceGeneratorSample


class RandomElasticDeformation:
    def __init__(self, kernel_sigma_choices: Tuple[Tuple[float, float, float, float]]):
        self.kernel_sigma_choices = kernel_sigma_choices

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        params = random.choice(self.kernel_sigma_choices)
        kernel = tuple(params[:2])
        sigma = tuple(params[2:])
        image_and_regions = torch.cat([sample.image, sample.regions])

        bilinear_et = K.augmentation.RandomElasticTransform(
            kernel_size=kernel, sigma=sigma, p=1.0
        )
        nearest_et = K.augmentation.RandomElasticTransform(
            kernel_size=kernel, sigma=sigma, p=1.0, mode="nearest"
        )

        params = bilinear_et.generate_parameters(sample.image.shape)
        sample.image = bilinear_et.apply_transform(sample.image, params)
        sample.regions = nearest_et.apply_transform(sample.regions, params)
        sample.hemispheres = nearest_et.apply_transform(sample.hemispheres, params)
        return sample
