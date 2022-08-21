import argparse
import json

import pytorch_lightning as pl

from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.model.train import BrainwaysDataModule
from brainways_reg_model.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--config", default="reg")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)

    # Load model
    model = BrainwaysRegModel.load_from_checkpoint(args.checkpoint)

    # init data
    synth_datamodule = BrainwaysDataModule(
        data_paths={
            "train": "data/synth.zip",
            "val": "data/synth.zip",
            "test": "data/synth.zip",
        },
        data_config=config.data,
        num_workers=args.num_workers,
        transform=model.transform,
        target_transform=model.target_transform,
    )

    real_datamodule = BrainwaysDataModule(
        data_paths={
            "train": "data/real.zip",
            "val": "data/real.zip",
            "test": "data/real.zip",
        },
        data_config=config.data,
        num_workers=args.num_workers,
        transform=model.transform,
        target_transform=model.target_transform,
    )

    # Initialize a trainer
    trainer = pl.Trainer(logger=False, gpus=1)

    # Test the model ⚡
    scores = trainer.test(
        model,
        dataloaders=[
            real_datamodule.test_dataloader(),
            synth_datamodule.test_dataloader(),
        ],
    )

    all_scores = {}
    for k, v in scores[0].items():
        all_scores[k.replace("dataloader_idx_0", "real")] = v
    for k, v in scores[1].items():
        all_scores[k.replace("dataloader_idx_1", "synth")] = v

    with open(args.output, "w") as fp:
        json.dump(all_scores, fp)


if __name__ == "__main__":
    main()
