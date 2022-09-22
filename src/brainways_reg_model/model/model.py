from functools import cached_property
from typing import Dict, List, Sequence, Union

import pytorch_lightning as pl
import torch
from kornia import augmentation as K
from PIL import Image
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection
from torchvision import models, transforms

from brainways_reg_model.model.approx_acc import ApproxAccuracy
from brainways_reg_model.model.atlas_registration_params import AtlasRegistrationParams
from brainways_reg_model.model.metric_dict_input_wrapper import MetricDictInputWrapper
from brainways_reg_model.utils.config import BrainwaysConfig
from brainways_reg_model.utils.data import model_label_to_value


class BrainwaysRegModel(pl.LightningModule):
    def __init__(self, config: BrainwaysConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.label_params = self.config.data.label_params

        self._build_model()
        self._build_metrics()

    def _build_metrics(self):
        metrics = MetricCollection(
            {
                "ap_acc_10": MetricDictInputWrapper(ApproxAccuracy(tolerance=10), "ap"),
                "ap_acc_20": MetricDictInputWrapper(ApproxAccuracy(tolerance=20), "ap"),
                "ap_mae": MetricDictInputWrapper(MeanAbsoluteError(), "ap"),
                "hem_acc": MetricDictInputWrapper(Accuracy(), "hemisphere"),
            }
        )
        if self.config.opt.train_confidence:
            metrics["confidence_acc"] = MetricDictInputWrapper(Accuracy(), "confidence")
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.config.model.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        num_outputs = sum(
            self.label_params[output].n_classes for output in self.config.model.outputs
        )
        # Confidence output
        num_outputs += 2

        _fc_layers = [
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.nll_loss

    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        outputs = {}
        out_i = 0
        for output_name in self.config.model.outputs:
            outputs[output_name] = x[
                :, out_i : self.label_params[output_name].n_classes
            ]
        confidence = x[:, -2:]
        return outputs, confidence

    def predict(
        self, x: Union[Image.Image, Sequence[Image.Image]]
    ) -> Union[AtlasRegistrationParams, List[AtlasRegistrationParams]]:
        if isinstance(x, Image.Image):
            x = [x]
            batch = False
        else:
            batch = True

        x = torch.stack([self.transform(image.convert("RGB")) for image in x]).to(
            self.device
        )
        y_logits, confidence = self(x)
        preds = self.postprocess(y_logits, confidence)

        if not batch:
            preds = preds[0]
        return preds

    def postprocess(
        self, logits: Dict[str, Tensor], confidence: Tensor
    ) -> List[AtlasRegistrationParams]:
        preds = {
            output_name: model_label_to_value(
                label=torch.argmax(logits[output_name], dim=-1),
                label_params=self.label_params[output_name],
            )
            for output_name in logits
        }
        confidence = torch.softmax(confidence, dim=-1)[:, 1].tolist()

        # Dict[List] -> List[Dict]
        list_of_dicts = [
            {output_name: float(preds[output_name][i]) for output_name in preds}
            for i in range(len(preds["ap"]))
        ]

        list_of_reg_params = [
            AtlasRegistrationParams(
                ap=d.get("ap"),
                rot_horizontal=d.get("rot_horizontal", 0.0),
                rot_sagittal=d.get("rot_sagittal", 0.0),
                hemisphere=self.label_params["hemisphere"].label_names[
                    int(d.get("hemisphere", 0))
                ],
                confidence=confidence[i],
            )
            for i, d in enumerate(list_of_dicts)
        ]

        return list_of_reg_params

    def step(self, batch, batch_idx, phase: str, metrics: MetricCollection):
        # TODO: refactor to incorporate "valid" and "confidence" more nicely
        # Forward pass
        y_logits, confidence = self(batch["image"])

        # Compute losses and metrics
        losses = {}
        pred_values = {}
        gt_values = {}
        for output_name in self.config.model.outputs:
            output_mask = batch[output_name + "_mask"]
            losses[output_name] = self.loss_func(
                input=F.log_softmax(y_logits[output_name][output_mask], dim=-1),
                target=batch[output_name][output_mask],
            )
            pred_values[output_name] = model_label_to_value(
                label=torch.argmax(y_logits[output_name][output_mask], dim=-1),
                label_params=self.label_params[output_name],
            )
            gt_values[output_name] = model_label_to_value(
                label=batch[output_name][output_mask],
                label_params=self.label_params[output_name],
            )

        # confidence loss
        # TODO: export to function?
        # need to re-calculate pred and gt to get non-valid values in their
        # correct place
        if self.config.opt.train_confidence:
            with torch.no_grad():
                pred_ap = model_label_to_value(
                    label=torch.argmax(y_logits["ap"], dim=-1),
                    label_params=self.label_params["ap"],
                ).int()
                gt_ap = model_label_to_value(
                    label=batch["ap"], label_params=self.label_params["ap"]
                ).int()
                confidence_label = (abs(gt_ap - pred_ap) < 20).int()
                confidence_label[~batch["valid"]] = 0

            pred_values["confidence"] = confidence
            gt_values["confidence"] = confidence_label

            confidence_loss = self.loss_func(
                input=F.log_softmax(confidence, dim=-1), target=confidence_label.long()
            )
            losses["confidence"] = confidence_loss

        metrics_ = metrics(pred_values, gt_values)
        self.log_dict(metrics_, prog_bar=True)

        loss = torch.mean(torch.stack(list(losses.values())))

        if phase != "train":
            self.log(f"{phase}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "val", self.val_metrics)

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "test", self.test_metrics)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(
            trainable_parameters,
            lr=self.config.opt.lr,
            weight_decay=self.config.opt.weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.config.opt.milestones,
            gamma=self.config.opt.lr_scheduler_gamma,
        )
        return [optimizer], [scheduler]

    @cached_property
    def target_transform(self):
        return transforms.Lambda(lambda x: torch.LongTensor(x))

    @cached_property
    def transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @cached_property
    def augmentation(self):
        return K.AugmentationSequential(
            # K.ColorJitter(0.2, 0.2, 0.2),
            K.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.7, 1.4),
                shear=(20.0, 20.0),
            ),
            # K.RandomPerspective(distortion_scale=0.5),
            K.RandomBoxBlur(p=0.2),
            K.RandomErasing(),
            K.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0)),
            K.RandomElasticTransform(kernel_size=(31, 31), sigma=(16.0, 16.0)),
            K.RandomElasticTransform(kernel_size=(21, 21), sigma=(12.0, 12.0)),
            random_apply=(1,),
            keepdim=True,
            return_transform=False,
            same_on_batch=False,
        )
