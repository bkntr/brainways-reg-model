from typing import Optional

import cv2
import napari
import numpy as np
from PIL import Image

from duracell.models.reg.model import BrainwaysRegModel
from duracell.utils.image import convert_to_uint8

model = BrainwaysRegModel.load_from_checkpoint("outputs/reg/real/model.ckpt")
model.freeze()

image = Image.open(
    "/home/ben/python/duracell/data/annotate/images/2021_helping_behavior/EF1.1_sl3_cFos_4_10_21.czi - Scene #0.jpg"
)

# image = preprocess(image)
pred = model.predict(image)
slice = model.slice_atlas(pred)
reference = slice["reference"]
# reference = crop_nonzero(reference)
reference = convert_to_uint8(reference)
reference = Image.fromarray(reference)

image = image.resize((256, 256))
reference = reference.resize((256, 256))

pts1 = [
    [0.0, 0.0],
    [0.0, image.width],
    [image.height, image.width],
    [image.height, 0.0],
]
pts2 = [
    [0.0, 0.0],
    [0.0, image.width],
    [image.height, image.width],
    [image.height, 0.0],
]

labels = ["image", "atlas"]

viewer = napari.Viewer()


def refresh_view(warp):
    top_row = np.concatenate([np.array(image), np.array(reference)], axis=1)
    top_row = np.tile(top_row[:, :, None], reps=(1, 1, 3))
    left_pad = image.width // 2
    right_pad = image.width - left_pad
    combined_view = np.zeros_like(image, shape=(image.height, image.width, 3))
    combined_view[:, :, 0] = np.array(reference)
    combined_view[:, :, 1] = np.array(warp)
    bot_row = np.concatenate(
        [
            np.zeros_like(combined_view, shape=(image.height, left_pad, 3)),
            combined_view,
            np.zeros_like(combined_view, shape=(image.height, right_pad, 3)),
        ],
        axis=1,
    )
    display_image = np.concatenate([top_row, bot_row], axis=0)
    if "view" in viewer.layers:
        viewer.layers["view"].data = display_image
    else:
        viewer.add_image(display_image, name="view")


def add_point():
    x1, y1 = points_layer.data[-1, 2], points_layer.data[-1, 1]
    x2, y2 = points_layer.data[-2, 2], points_layer.data[-2, 1]

    if y1 > image.height:
        x1 -= image.width // 2
        y1 -= image.height
        x2 -= image.width // 2
        y2 -= image.height
    else:
        x2 -= image.width

    pts1.append([x1, y1])
    pts2.append([x2, y2])


@viewer.bind_key("Backspace")
def remove_point(viewer: Optional[napari.Viewer] = None):
    if len(pts1) > 4:
        points_layer.properties
        points_layer.data = points_layer.data[:-2]
        pts1.pop(-1)
        pts2.pop(-1)
        calc_transform()


def calc_transform():
    pts1_ = np.array(pts1)[None, ...]
    pts2_ = np.array(pts2)[None, ...]

    matches = list()
    for ipoint in range(0, pts1_.shape[1]):
        matches.append(cv2.DMatch(ipoint, ipoint, 0))
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(pts2_, pts1_, matches)
    out_img = tps.warpImage(np.array(image))
    refresh_view(out_img)


def next_label(event=None):
    """Keybinding to advance to the next label with wraparound"""

    # get the currently selected label
    current_properties = points_layer.current_properties
    current_label = current_properties["label"][0]

    # determine the index of that label in the labels list
    ind = list(labels).index(current_label)

    # increment the label with wraparound
    new_ind = (ind + 1) % len(labels)

    # get the new label and assign it
    new_label = labels[new_ind]
    current_properties["label"] = np.array([new_label])
    points_layer.current_properties = current_properties


def on_click(event):
    """Mouse click binding to advance the label when a point is added"""
    # only do something if we are adding points
    if event.type == "set_data":
        if len(points_layer.data) and len(points_layer.data) % 2 == 0:
            add_point()
            calc_transform()
            # points_layer.data = np.zeros(shape=(0, 3), dtype=np.float64)

        next_label()

        # by default, napari selects the point that was just added
        # disable that behavior, as the highlight gets in the way
        points_layer.selected_data = []


COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
]

refresh_view(image)

points_layer = viewer.add_points(
    properties={"label": labels},
    edge_color="label",
    edge_color_cycle=COLOR_CYCLE,
    symbol="x",
    face_color="label",
    edge_width=1,
    size=5,
    ndim=3,
)

points_layer.mode = "add"
points_layer.events.connect(on_click)

napari.run()
