from typing import Iterable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Sample = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def train_step(teacher_model: nn.Module,
               student_model: nn.Module,
               im: torch.Tensor,
               masked_im: torch.Tensor,
               bool_mask: torch.Tensor,
               k: int = 3,
               loss_scale: float = 1.0,
               beta: float = 1.0) -> float:
    """Single training step.

    Parameters
    ----------
    model : nn.Module
        Data2Vec model.
    im : torch.Tensor
        Input image for the teacher model to generate the  targets. 
        Tensor shape [N, T, H]
    masked_im : torch.Tensor
        Randomly masked image. Input for the student model. 
        Tensor shape [N, T, H]
    k : int
        Regress the last k hidden states average. Defaults 3.
    beta : float
        Beta for the smooth l1 loss. Defaults 1.0.

    Returns
    -------
    float
        Loss value
    """
    y = _generate_targets(teacher_model, im, k=k)
    x, _ = student_model(masked_im)

    x = x[bool_mask.long()]
    y = y[bool_mask.long()]

    loss = F.smooth_l1_loss(x, y, reduction="none", beta=beta)
    loss = loss.sum(dim=2).mean()
    (loss * loss_scale).backward()

    return loss.item()


def params_ema_update(teacher_model: nn.Module, student_model: nn.Module,
                      tau: float) -> None:
    new_params = student_model.named_parameters()
    target_params = teacher_model.named_parameters()
    for (n, p), (n1, p1) in zip(target_params, new_params):
        if n != n1:
            raise ValueError(f"Unexpected parameter name: {n1}")
        p.data.copy_(tau * p.data + (1 - tau) * p1.data)


def tau_generator(*,
                  initial_step: int = 0,
                  min_value: float = .2,
                  max_value: float = .7,
                  increase_steps: int = 512) -> Iterator[float]:
    step = initial_step
    slope = (max_value - min_value) / increase_steps
    while True:
        if step < increase_steps:
            yield step * slope + min_value
            step += 1
        else:
            yield max_value


def train_single_epoch(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dl: Iterable[Sample],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    tau_gen: Iterator[float],
    epoch: int,
    k: int = 3,
    beta: float = 1.0,
    accum_steps: int = 1,
    device: torch.device = torch.device("cpu")) -> None:
    """Executes a trining epoch.

    Parameters
    ----------
    teacher_model : nn.Module
        Data2Vec model in teacher mode.
    student_model : nn.Module
        Data2Vec model in student mode.
    dl : Iterator[Sample]
        Iterator to go through the training set. 
    optimizer : torch.optim.Optimizer
        Student model optimizer.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler. The  `step` method is called on every training
        step.
    tau_gen : Iterator[float]
        Generator to retrieve the tau for smoothly update the teacher model.
    epoch : int
        Current epoch number. Just for logging purposes.
    k : int
        Regress the last k hidden states average. Defaults 3.
    beta : float
        Beta for the smooth l1 loss. Defaults 1.0.
    accum_steps : int
        Gradient accumulation steps. Defaults 1.
    device : torch.device
        Device where the train should run on. Defaults torch.device("cpu").
    """
    running_loss = 0.0
    student_model.train()
    optimizer.zero_grad()

    for i, (im, (masked_im, bool_mask)) in enumerate(dl, start=1):
        im = im.to(device)
        masked_im = masked_im.to(device)
        bool_mask = bool_mask.to(device)

        running_loss += train_step(teacher_model,
                                   student_model,
                                   im,
                                   masked_im,
                                   bool_mask,
                                   k=k,
                                   beta=beta,
                                   loss_scale=1.0 / accum_steps)
        nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)

        if i % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            tau = next(tau_gen)
            params_ema_update(teacher_model, student_model, tau)

        if i % 30 * accum_steps == 0:
            loss_mean = running_loss / i
            [lr] = lr_scheduler.get_last_lr()
            print(f"Epoch [{epoch}] loss: {loss_mean:.4f}",
                  f"tau: {tau:.4f}",
                  f"learning-rate: {lr:.2e}",
                  sep="  ")

    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def _generate_targets(model: nn.Module,
                      inputs: torch.Tensor,
                      k: int = 4) -> torch.Tensor:
    # Generate the targets in teacher mode
    model.eval()
    _, hidden_states = model(inputs)
    top_k_hs = hidden_states[-k:]  # [K, N, L, C]

    # Prepare tensor for instance_norm function
    k, n, l, c = top_k_hs.size()
    top_k_hs = top_k_hs.view(-1, l, c)  # [K * N, L, C]
    top_k_hs = top_k_hs.permute(0, 2, 1)  # [K * N, C, L]
    top_k_hs = F.instance_norm(top_k_hs)

    # Get back to expected shape
    top_k_hs = top_k_hs.view(k, n, c, l)  # [K, N, C, L]
    top_k_hs = top_k_hs.permute(0, 1, 3, 2)  # [K, N, L, C]

    # Average the K blocks
    return top_k_hs.mean(0)
