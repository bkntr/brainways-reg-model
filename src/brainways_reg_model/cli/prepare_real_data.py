import shutil
from pathlib import Path
import tempfile
from typing import List

import pandas as pd
import yaml
from bg_atlasapi import BrainGlobeAtlas


def prepare_real_data(
    phase: str,
    metadata,
    annotations: pd.DataFrame,
    output_dir: Path,
):
    output_dir = output_dir / phase
    output_dir.mkdir()

    # write metadata
    with open(output_dir / "metadata.yaml", "w") as outfile:
        yaml.dump(metadata, outfile, default_flow_style=False)

    image_paths = annotations.index
    annotations.index = [Path(x).name for x in image_paths]

    # write labels
    annotations.to_csv(output_dir / "labels.csv")

    # write images
    images_root_path = output_dir / "images"
    images_root_path.mkdir()
    for image_path in image_paths:
        src = Path("data/annotate/images") / image_path
        dst = images_root_path / Path(image_path).name
        assert src.exists()
        assert not dst.exists()
        shutil.copy(src, dst)


def main():
    tmp_dir = Path(tempfile.mkdtemp())

    # write metadata
    atlas = BrainGlobeAtlas("whs_sd_rat_39um")
    metadata = {
        "atlas": atlas.atlas_name,
        "ap_size": atlas.shape[0],
        "si_size": atlas.shape[1],
        "lr_size": atlas.shape[2],
    }

    # write labels
    annotations = pd.read_csv("data/annotate/omer_itz_annotations.csv", index_col=0)
    if "image_path" in annotations.columns:
        annotations.drop(columns="image_path", inplace=True)

    animal_id = annotations.index.str.extract(r".*(Dev\d+).*", expand=False)
    test_animal_ids = ["Dev24", "Dev25", "Dev26"]
    val_animal_ids = ["Dev27", "Dev28"]
    test_annotations = annotations.loc[animal_id.isin(test_animal_ids)]
    val_annotations = annotations.loc[animal_id.isin(val_animal_ids)]
    train_annotations = annotations.loc[
        ~animal_id.isin(test_animal_ids + val_animal_ids)
    ]

    prepare_real_data(
        phase="test",
        metadata=metadata,
        annotations=test_annotations,
        output_dir=tmp_dir,
    )

    prepare_real_data(
        phase="val",
        metadata=metadata,
        annotations=val_annotations,
        output_dir=tmp_dir,
    )
    prepare_real_data(
        phase="train",
        metadata=metadata,
        annotations=train_annotations,
        output_dir=tmp_dir,
    )

    Path("data/real.zip").unlink(missing_ok=True)
    shutil.make_archive("data/real", "zip", tmp_dir)
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    main()
