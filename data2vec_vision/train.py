import random
from os import PathLike
from pathlib import Path
from typing import Optional

import click
import torch

from data2vec_vision import Data2Vec, Data2VecDataset
from data2vec_vision.engine import tau_generator, train_single_epoch
from data2vec_vision.transforms import PatchesToSequence, build_train_tfms


@click.command()
# Input data related parameters
@click.argument("input-path", type=click.Path(file_okay=False, exists=True))
@click.option("--image-size",
              type=int,
              default=224,
              help="Initial image size, before spliting it into patches.")
@click.option("--patch-size",
              type=int,
              default=16,
              help="Images will be splitted in patches of this size.")
# Training duration
@click.option("--epochs",
              type=int,
              default=200,
              help="Number of training epochs.")
@click.option("--batch-size",
              type=int,
              default=32,
              help="Training batch size.")
# ViT model hyper parameters
@click.option("--n-encoders",
              type=int,
              default=8,
              help="ViT model number of encoders.")
@click.option("--dropout",
              type=float,
              default=.2,
              help="Encoder layers dopout rate")
@click.option("--hidden-dim",
              type=int,
              default=768,
              help="ViT hidden number of dimensions.")
@click.option("--mask-erase-pct",
              type=float,
              default=.6,
              help="Percentage of the image to delete.")
# Optimization Hyperparameters
@click.option("--max-lr",
              type=float,
              default=1e-1,
              help="One cycle lr scheduler max learning rate.")
@click.option("--k",
              type=int,
              default=4,
              help="Number of top hidden states to regress.")
@click.option("--grad-accum",
              type=int,
              default=1,
              help="Gradient accumulation steps.")
@click.option("--beta", type=float, default=2., help="L1 smooth loss beta.")
# Teacher model EMA update
@click.option("--tau-initial",
              type=float,
              default=0.1,
              help="Teacher model EMA minimum update factor.")
@click.option("--tau-end",
              type=float,
              default=0.99,
              help="Teacher model EMA maximum update factor.")
@click.option("--tau-increase-pct",
              type=float,
              default=0.3,
              help="Teacher model EMA factor will increase till "
              "`tau-increase-pct` * `epochs` then will stay constant.")
# Checkpointing parameters
@click.option("--checkpoint-out-dir",
              type=click.Path(file_okay=False),
              default="models/")
@click.option("--resume-checkpoint",
              type=click.Path(exists=True, dir_okay=False),
              default=None)
def train(
    input_path: PathLike,
    image_size: int,
    patch_size: int,
    epochs: int,
    batch_size: int,
    n_encoders: int,
    dropout: float,
    hidden_dim: int,
    mask_erase_pct: float,
    max_lr: float,
    grad_accum: int,
    k: int,
    beta: float,
    tau_initial: float,
    tau_end: float,
    tau_increase_pct: float,
    checkpoint_out_dir: PathLike,
    resume_checkpoint: Optional[str],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = torch.Generator()
    g.manual_seed(0)
    torch.manual_seed(0)
    random.seed(0)

    checkpoint_out_dir = Path(checkpoint_out_dir)
    checkpoint_out_dir.mkdir(exist_ok=True, parents=True)

    input_path = Path(input_path)
    paths = list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))

    train_tfm, masking_fn = build_train_tfms((image_size, image_size),
                                             patch_size=patch_size,
                                             mask_erase_pct=mask_erase_pct)
    train_ds = Data2VecDataset(paths,
                               masking_fn=masking_fn,
                               before_masking_tfms=train_tfm,
                               after_masking_tfms=PatchesToSequence())

    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           generator=g)

    steps_per_epoch = len(train_dl) // grad_accum
    student_model = Data2Vec(in_features=patch_size * patch_size * 3,
                             seq_len=(image_size // patch_size)**2,
                             dropout=dropout,
                             n_encoders=n_encoders,
                             dim=hidden_dim)
    student_model.to(device)

    teacher_model = Data2Vec(in_features=patch_size * patch_size * 3,
                             seq_len=(image_size // patch_size)**2,
                             dropout=dropout,
                             n_encoders=n_encoders,
                             dim=hidden_dim)
    teacher_model.to(device)
    teacher_model.load_state_dict(student_model.state_dict())  # type: ignore

    optimizer = torch.optim.SGD(params=student_model.parameters(),
                                lr=1e-3,
                                weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        anneal_strategy="cos")

    tau_increase_steps = int(epochs * tau_increase_pct) * steps_per_epoch
    if resume_checkpoint is not None:
        chkp = torch.load(resume_checkpoint)
        teacher_model.load_state_dict(chkp["teacher"])
        student_model.load_state_dict(chkp["student"])
        scheduler.load_state_dict(chkp["lr_scheduler"])
        optimizer.load_state_dict(chkp["optimizer"])
        tau_gen = tau_generator(initial_step=chkp["step"],
                                min_value=tau_initial,
                                max_value=tau_end,
                                increase_steps=tau_increase_steps)
        start_epoch = chkp["step"] // steps_per_epoch
    else:
        tau_gen = tau_generator(min_value=tau_initial,
                                max_value=tau_end,
                                increase_steps=tau_increase_steps)
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_single_epoch(teacher_model,
                           student_model,
                           dl=train_dl,
                           optimizer=optimizer,
                           lr_scheduler=scheduler,
                           tau_gen=tau_gen,
                           epoch=epoch,
                           device=device,
                           k=k,
                           beta=beta,
                           accum_steps=grad_accum)

        if (epoch + 1) % 10 == 0:
            chkp = {
                "student": student_model.state_dict(),
                "teacher": teacher_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "step": (epoch + 1) * steps_per_epoch,
            }
            torch.save(chkp, checkpoint_out_dir / f"checkpoint_{epoch}.pt")

            student_model.save(checkpoint_out_dir / f"student_{epoch}.pt")
