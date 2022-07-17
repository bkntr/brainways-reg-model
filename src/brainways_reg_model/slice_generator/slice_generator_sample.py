from dataclasses import dataclass
from typing import Tuple, Optional, Union

import torch
from PIL import Image


@dataclass
class SliceGeneratorSample:
    ap: float = 0
    si: float = 0
    lr: float = 0
    rot_frontal: float = 0
    rot_horizontal: float = 0
    rot_sagittal: float = 0
    hemisphere: str = "both"
    scale: Tuple[float, float] = (1.0, 1.0)
    image: Optional[Union[torch.Tensor, Image.Image]] = None
    regions: Optional[Union[torch.Tensor, Image.Image]] = None
    hemispheres: Optional[Union[torch.Tensor, Image.Image]] = None
