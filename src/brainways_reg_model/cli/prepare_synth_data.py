import argparse
import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from aicsimageio.readers import TiffReader
from bg_atlasapi import BrainGlobeAtlas
from pytorch_lightning.utilities.seed import pl_worker_init_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from duracell.slice_generator.slice_generator import SliceGenerator
from duracell.slice_generator.slice_generator_sample import SliceGeneratorSample
from duracell.utils.config import load_config, load_synth_config


def atlas_reference(atlas: str, brainglobe: bool, axes: Tuple[int, int, int]):
    if brainglobe:
        atlas = BrainGlobeAtlas(atlas)
        reference = atlas.reference
        reference = reference.transpose(axes)
    else:
        reference = TiffReader(atlas).data
        reference = reference.transpose(axes)
        reference = reference[228:1130]
        reference = reference.astype(float)
        reference = (
            (reference - reference.min()) / (reference.max() - reference.min()) * 255
        ).astype(np.uint8)
        reference[reference < 50] = 0
    return reference


def create_dataset(
    phase: str,
    atlas: BrainGlobeAtlas,
    stages: List[Dict],
    n: int,
    rot_frontal_limit: Tuple[float, float],
    rot_horizontal_limit: Tuple[float, float],
    rot_sagittal_limit: Tuple[float, float],
    output: Path,
):
    root_dir = output / phase
    images_dir = root_dir / "images"
    images_dir.mkdir(parents=True)

    generator = DataLoader(
        SliceGenerator(
            atlas=atlas,
            stages=stages,
            n=n,
            rot_frontal_limit=rot_frontal_limit,
            rot_horizontal_limit=rot_horizontal_limit,
            rot_sagittal_limit=rot_sagittal_limit,
        ),
        batch_size=None,
        num_workers=16,
        worker_init_fn=pl_worker_init_function,
    )
    all_labels = []
    for sample_idx, sample in enumerate(tqdm(generator, desc=phase)):
        sample: SliceGeneratorSample
        sample.image.save(images_dir / f"{sample_idx}.jpg")
        sample.regions.save(
            str(images_dir / f"{sample_idx}-structures.tif"), compression="tiff_lzw"
        )

        # save non-image parameters
        attrs = asdict(sample)
        del attrs["image"]
        del attrs["regions"]
        del attrs["hemispheres"]
        all_labels.append(attrs)
    all_labels = pd.DataFrame(all_labels)
    all_labels.to_csv(str(root_dir / "labels.csv"), float_format="%.3f")

    metadata = {
        "atlas": atlas.atlas_name,
        "ap_size": atlas.shape[0],
        "si_size": atlas.shape[1],
        "lr_size": atlas.shape[2],
        "rot_frontal_limit": list(rot_frontal_limit),
        "rot_horizontal_limit": list(rot_horizontal_limit),
        "rot_sagittal_limit": list(rot_sagittal_limit),
    }

    with open(root_dir / "metadata.yaml", "w") as outfile:
        yaml.dump(metadata, outfile, default_flow_style=False)


def prepare_data(phase: str, output_dir: Path):
    config = load_config()
    synth_config = load_synth_config()
    atlas = BrainGlobeAtlas(config.data.atlas.name)

    create_dataset(
        phase=phase,
        atlas=atlas,
        stages=synth_config["stages"],
        n=synth_config[phase],
        rot_frontal_limit=config.data.label_params["rot_frontal"].limits,
        rot_horizontal_limit=config.data.label_params["rot_horizontal"].limits,
        rot_sagittal_limit=config.data.label_params["rot_sagittal"].limits,
        output=output_dir,
    )


def main():
    pl.seed_everything(load_config().seed)

    if Path("data/synth").exists():
        if input("Synthetic data already exists, overwrite? [y/N] ")[0].lower() == "y":
            shutil.rmtree("data/synth")
        else:
            return

    tmp_dir = Path(f"/tmp/duracell/synth/")
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    tmp_dir.mkdir(parents=True)

    for phase in ("test", "val", "train"):
        prepare_data(phase, tmp_dir)

    Path("data").mkdir(exist_ok=True)
    Path("data/synth.zip").unlink(missing_ok=True)
    shutil.make_archive("data/synth", "zip", tmp_dir)
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    main()
