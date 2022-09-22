import shutil
import tempfile
from pathlib import Path

import click
import pandas as pd
import yaml

from brainways_reg_model.utils.paths import (
    REAL_DATA_ROOT,
    REAL_DATA_ZIP_PATH,
    REAL_RAW_DATA_ROOT,
)


def prepare_real_data_phase(
    phase: str,
    metadata,
    labels: pd.DataFrame,
    output_dir: Path,
):
    output_dir = output_dir / phase
    output_dir.mkdir()

    # write metadata
    with open(output_dir / "metadata.yaml", "w") as outfile:
        yaml.dump(metadata, outfile, default_flow_style=False)

    # write labels
    labels.to_csv(output_dir / "labels.csv", index=False)

    # write images
    input_images_root = REAL_RAW_DATA_ROOT / "images"
    output_images_root = output_dir / "images"
    output_images_root.mkdir()
    for image_path in labels.filename.to_list():
        src = input_images_root / image_path
        dst = output_images_root / image_path
        dst.parent.mkdir(exist_ok=True, parents=True)
        assert src.exists()
        assert not dst.exists()
        shutil.copy(src, dst)


@click.command()
def prepare_real_data():
    tmp_dir = Path(tempfile.mkdtemp())
    labels = pd.read_csv(REAL_RAW_DATA_ROOT / "labels.csv")
    with open(REAL_RAW_DATA_ROOT / "metadata.yaml") as fd:
        metadata = yaml.safe_load(fd)
    test_animal_ids = ["Dev24", "Dev25", "81-1", "Retro2", "29-2"]
    val_animal_ids = ["Dev27", "Dev28", "Retro1", "85-2"]
    test_labels = labels.loc[labels.animal_id.isin(test_animal_ids)]
    val_labels = labels.loc[labels.animal_id.isin(val_animal_ids)]
    train_labels = labels.loc[~labels.animal_id.isin(test_animal_ids + val_animal_ids)]

    prepare_real_data_phase(
        phase="test",
        metadata=metadata,
        labels=test_labels,
        output_dir=tmp_dir,
    )

    prepare_real_data_phase(
        phase="val",
        metadata=metadata,
        labels=val_labels,
        output_dir=tmp_dir,
    )
    prepare_real_data_phase(
        phase="train",
        metadata=metadata,
        labels=train_labels,
        output_dir=tmp_dir,
    )

    Path(REAL_DATA_ZIP_PATH).unlink(missing_ok=True)
    shutil.rmtree(REAL_DATA_ROOT)
    shutil.make_archive(REAL_DATA_ZIP_PATH.with_suffix(""), "zip", tmp_dir)
    shutil.rmtree(str(tmp_dir))
