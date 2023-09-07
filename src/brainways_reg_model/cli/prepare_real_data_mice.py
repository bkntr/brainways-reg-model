import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from brainways_reg_model.utils.paths import (
    REAL_DATA_ROOT_MICE,
    REAL_DATA_ZIP_PATH_MICE,
    REAL_RAW_DATA_ROOT_MICE,
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
    input_images_root = REAL_RAW_DATA_ROOT_MICE / "images"
    output_images_root = output_dir / "images"
    output_images_root.mkdir()
    for image_path in tqdm(labels.filename.to_list(), desc=phase):
        src = input_images_root / image_path
        dst = output_images_root / image_path
        dst.parent.mkdir(exist_ok=True, parents=True)
        assert src.exists()
        assert not dst.exists()
        shutil.copy(src, dst)


@click.command()
def prepare_real_data_mice():
    tmp_dir = Path(tempfile.mkdtemp())
    labels = pd.read_csv(REAL_RAW_DATA_ROOT_MICE / "labels.csv")
    with open(REAL_RAW_DATA_ROOT_MICE / "metadata.yaml") as fd:
        metadata = yaml.safe_load(fd)

    all_indices = np.random.permutation(len(labels))
    train_indices = all_indices[: int(len(all_indices) * 0.5)]
    val_indices = all_indices[
        int(len(all_indices) * 0.5) : int(len(all_indices) * 0.75)
    ]
    test_indices = all_indices[int(len(all_indices) * 0.75) :]

    test_labels = labels.iloc[test_indices]
    val_labels = labels.iloc[val_indices]
    train_labels = labels.iloc[train_indices]

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

    Path(REAL_DATA_ZIP_PATH_MICE).unlink(missing_ok=True)
    shutil.rmtree(REAL_DATA_ROOT_MICE, ignore_errors=True)
    shutil.make_archive(REAL_DATA_ZIP_PATH_MICE.with_suffix(""), "zip", tmp_dir)
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    prepare_real_data_mice()
