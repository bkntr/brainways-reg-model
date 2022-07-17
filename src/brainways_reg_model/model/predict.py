import argparse
import random
from pathlib import Path
from typing import Optional

import PIL.Image
import napari
import numpy as np
from magicgui import magicgui
from napari.types import ImageData

from duracell.models.reg.model import BrainwaysRegModel
from duracell.ui.utils import update_layer_contrast_limits
from duracell.utils.image import nonzero_bounding_box, brain_mask


class RegistrationAnnotator:
    def __init__(
        self,
        images_root: str,
        filter: Optional[str] = None,
    ):
        self.image_paths = list(Path(images_root).glob(filter or "*"))
        self.current_image_idx = 0

        self.viewer = napari.Viewer()
        self.model = BrainwaysRegModel.load_from_checkpoint(
            "outputs/reg/real/model.ckpt"
        )
        self.model.freeze()

        self._input_translate = (0, self.model.atlas.shape[2])
        self._overlay_translate = (
            self.model.atlas.shape[1],
            self.model.atlas.shape[2] // 2,
        )

        self.input_layer = self.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Input",
        )
        self.atlas_slice_layer = self.viewer.add_image(
            np.zeros((self.model.atlas.shape[1], self.model.atlas.shape[2]), np.uint8),
            name="Atlas Slice",
        )
        self.input_layer.translate = self._input_translate

        self.registration_params_widget = magicgui(
            self.registration_params,
            auto_call=True,
            ap={
                "label": "Anterior-Posterior",
                "widget_type": "FloatSlider",
                "max": self.model.atlas.shape[0] - 1,
                "enabled": False,
            },
            rot_horizontal={
                "label": "Horizontal Rotation",
                "widget_type": "FloatSlider",
                "min": -15,
                "max": 15,
                "enabled": False,
            },
            hemisphere={
                "label": "Hemisphere",
                "widget_type": "RadioButtons",
                "orientation": "horizontal",
                "choices": [("Left", "left"), ("Both", "both"), ("Right", "right")],
            },
            confidence={
                "label": "Confidence",
                "widget_type": "FloatSlider",
                "min": 0,
                "max": 1,
                "step": 0.01,
            },
        )

        self.image_slider_widget = magicgui(
            self.image_slider,
            auto_call=True,
            image_number={
                "widget_type": "Slider",
                "label": "Image #",
                "min": 1,
                "max": len(self.image_paths),
            },
        )

        self.viewer.window.add_dock_widget(
            self.registration_params_widget, name="Annotate", area="right"
        )

        self.viewer.window.add_dock_widget(
            self.image_slider_widget, name="Images", area="right"
        )

        self.change_image()

        self.viewer.bind_key("r", self.random_image)

    def random_image(self, viewer: napari.Viewer):
        self.current_image_idx = random.randint(0, len(self.image_paths) - 1)
        self.change_image()

    def registration_params(
        self,
        ap: float,
        rot_horizontal: float,
        hemisphere: str,
        confidence: float,
    ) -> napari.types.LayerDataTuple:
        pass

    def image_slider(self, image_number: int):
        image_idx = image_number - 1
        if self.current_image_idx != image_idx:
            self.current_image_idx = image_idx
            self.change_image()

    @property
    def image_path(self) -> str:
        return self.image_paths[self.current_image_idx].as_posix()

    def change_image(self):
        image = PIL.Image.open(self.image_path)
        brain_box = nonzero_bounding_box(brain_mask(np.array(image)))
        params = self.model.predict(image)

        self.registration_params_widget(
            ap=params.ap,
            rot_horizontal=params.rot_horizontal,
            hemisphere=params.hemisphere,
            confidence=params.confidence,
            update_widget=True,
        )
        atlas_slice = self.model.atlas.slice(
            ap=params.ap,
            rot_horizontal=params.rot_horizontal,
            hemisphere=params.hemisphere,
        ).reference.numpy()

        self.input_layer.data = np.array(image)
        update_layer_contrast_limits(self.input_layer)
        self.atlas_slice_layer.data = atlas_slice
        update_layer_contrast_limits(self.atlas_slice_layer)
        atlas_box = self.model.atlas.bounding_boxes[int(params.ap)]
        input_scale = atlas_box[3] / brain_box[3]
        self.input_layer.scale = (input_scale, input_scale)
        tx = (brain_box[0] + brain_box[0] * 0.1) * input_scale
        ty = brain_box[1] - atlas_box[1]
        self.atlas_slice_layer.translate = (ty, tx)
        self.viewer.reset_view()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/real/test/images")
    args = parser.parse_args()

    annotator = RegistrationAnnotator(images_root=args.images)
    napari.run()


if __name__ == "__main__":
    main()
