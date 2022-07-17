from kornia.utils import ImageToTensor

from duracell.slice_generator.stages.adjust_contrast import AdjustContrast
from duracell.slice_generator.stages.crop_material_area import CropMaterialArea
from duracell.slice_generator.stages.filter_regions import FilterRegions
from duracell.slice_generator.stages.random_elastic_deformation import (
    RandomElasticDeformation,
)
from duracell.slice_generator.stages.random_affine import RandomAffine
from duracell.slice_generator.stages.random_light_deformation import (
    RandomLightDeformation,
)
from duracell.slice_generator.stages.random_lighten_dark_areas import (
    RandomLightenDarkAreas,
)
from duracell.slice_generator.stages.random_mask_regions import RandomMaskRegions
from duracell.slice_generator.stages.random_single_hemisphere import (
    RandomSingleHemisphere,
)
from duracell.slice_generator.stages.random_zero_below_threshold import (
    RandomZeroBelowThreshold,
)
from duracell.slice_generator.stages.resize import Resize
from duracell.slice_generator.stages.to_kornia import ToKornia
from duracell.slice_generator.stages.to_pil_image import ToPILImage

stages_dict = {
    "crop_material_area": CropMaterialArea,
    "to_tensor": ImageToTensor,
    "random_affine": RandomAffine,
    "random_elastic_deformation": RandomElasticDeformation,
    "random_zero_below_threshold": RandomZeroBelowThreshold,
    "random_mask_regions": RandomMaskRegions,
    "random_lighten_dark_areas": RandomLightenDarkAreas,
    "random_light_deformation": RandomLightDeformation,
    "adjust_contrast": AdjustContrast,
    "resize": Resize,
    "to_kornia": ToKornia,
    "to_pil_image": ToPILImage,
    "random_single_hemisphere": RandomSingleHemisphere,
    "filter_regions": FilterRegions,
}
