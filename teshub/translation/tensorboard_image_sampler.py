# mypy: disable-error-code="misc, no-any-unimported"
from dataclasses import dataclass

import lightning.pytorch as pl  # type: ignore[import]
import torch
import torchvision  # type: ignore[import]
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize  # type: ignore[import]


@dataclass(eq=False)
class TensorboardImageSampler(pl.callbacks.Callback):
    visualize_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    use_att: bool
    use_seg: bool
    device: torch.device = (
        torch.device('cpu') if torch.cuda.is_available()
        else torch.device('cuda')
    )

    def _revert_normalization(self, img: torch.Tensor) -> torch.Tensor:
        reverted: torch.Tensor = normalize(
            img, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        reverted = normalize(reverted, (-0.5, -0.5, -0.5), (1., 1., 1.))

        return reverted

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        generated_images: torch.Tensor | None = None
        source_images: torch.Tensor | None = None

        with torch.no_grad():
            pl_module.eval()
            for batch in self.visualize_dataloader:
                source_data, _ = batch
                source_image_data = source_data[0].to(self.device)

                generated_image_batch: torch.Tensor
                source_image_batch: torch.Tensor
                if self.use_seg and not self.use_att:
                    raise RuntimeError(
                        "Can't have use_seg=True and use_att=False")

                if self.use_att and self.use_seg:
                    source_image_batch, attention_batch, _ = (
                        torch.split(source_image_data, [3, 1, 1], dim=1)
                    )

                    generated_image_batch = pl_module(
                        source_image_batch, attention_batch)
                elif self.use_att and not self.use_seg:
                    source_image_batch, attention_batch = (
                        torch.split(source_image_data, [3, 1], dim=1)
                    )

                    generated_image_batch = pl_module(
                        source_image_batch, attention_batch)
                else:
                    source_image_batch = source_image_data
                    generated_image_batch = pl_module(source_image_batch)

                generated_images = (
                    generated_image_batch
                    if generated_images is None
                    else torch.cat(
                        [generated_images, generated_image_batch], dim=0)
                )

                source_images = (
                    source_image_batch
                    if source_images is None
                    else torch.cat(
                        [source_images, source_image_batch], dim=0)
                )
            pl_module.train()

        assert generated_images is not None
        assert source_images is not None

        images = self._revert_normalization(
            torch.cat([source_images, generated_images], dim=0)
        )

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=len(self.visualize_dataloader)
        )

        str_title = f"{pl_module.__class__.__name__}_images"
        # This doesnt require conversion to RGB by multiplying with 255
        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step)
