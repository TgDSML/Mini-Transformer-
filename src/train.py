from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.config import get_config, latest_weights_path, weights_path
from src.data.text import make_loaders
from src.models.transformer_lm import TransformerEncoderLM


def build_model(cfg: dict[str, Any], vocab_size: int) -> TransformerEncoderLM:
    return TransformerEncoderLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        pos_encoding=cfg["positional_encoding"],
    )


def build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(
    cfg: dict[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int
) -> tuple[SequentialLR, int, int]:
    total_steps = cfg["num_epochs"] * steps_per_epoch
    warmup_steps = min(
        total_steps - 1,
        cfg.get("warmup_steps", max(100, int(0.05 * total_steps))),
    )
    warmup_steps = max(0, warmup_steps)
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=cfg.get("warmup_start_factor", 0.01),
        total_iters=max(1, warmup_steps),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=cfg.get("min_lr", 1e-5),
    )

    if warmup_steps == 0:
        scheduler = SequentialLR(optimizer, [cosine], milestones=[])
    else:
        scheduler = SequentialLR(
            optimizer,
            [warmup, cosine],
            milestones=[warmup_steps],
        )

    return scheduler, total_steps, warmup_steps


def save_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        ckpt_path,
    )


def load_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location=device if device is not None else "cpu",
    )
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def resolve_resume_path(cfg: dict[str, Any]) -> str | None:
    resume_from = cfg.get("resume_from")
    if resume_from:
        return str(resume_from)
    if cfg.get("resume_latest"):
        return latest_weights_path(cfg)
    return None


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


@torch.no_grad()
def sample_text(
    model: TransformerEncoderLM,
    tokenizer: Any,
    device: torch.device,
    prompt: str = " ",
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    if not prompt:
        prompt = " "

    safe_prompt = "".join(ch for ch in prompt if ch in tokenizer.stoi)
    if not safe_prompt:
        safe_prompt = tokenizer.chars[0]

    idx = torch.tensor([tokenizer.encode(safe_prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())


def train_model(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(get_config() if cfg is None else cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = str(Path("data") / cfg["data_file"])

    (
        tokenizer,
        _ids,
        _train_ids,
        _val_ids,
        train_loader,
        val_loader,
        _train_ds,
        _val_ds,
        train_batches,
        _val_batches,
        _n_tokens,
        _n_train_tokens,
    ) = make_loaders(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        path_=data_path,
        split=cfg["train_val_split"],
        shuffle=cfg["shuffle"],
    )

    model = build_model(cfg, tokenizer.vocab_size()).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler, total_steps, _warmup_steps = build_scheduler(
        cfg,
        optimizer,
        max(1, train_batches),
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    resume_path = resolve_resume_path(cfg)
    if resume_path:
        checkpoint = load_ckpt(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed_from={resume_path} start_epoch={start_epoch} global_step={global_step}")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            batch_count += 1

            if global_step >= cfg.get("max_steps", total_steps):
                break

        train_loss = epoch_loss / max(1, batch_count)
        val_loss = evaluate(model, val_loader, device)
        best_val_loss = min(best_val_loss, val_loss)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if epoch % cfg["save_every"] == 0:
            save_ckpt(
                weights_path(cfg, epoch),
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} step={global_step} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f}"
        )

        if global_step >= cfg.get("max_steps", total_steps):
            break

    return {
        "device": str(device),
        "epochs_ran": len(history),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def train_and_sample(
    cfg: dict[str, Any] | None = None,
    prompt: str = " ",
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    cfg = dict(get_config() if cfg is None else cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = str(Path("data") / cfg["data_file"])
    (
        tokenizer,
        _ids,
        _train_ids,
        _val_ids,
        train_loader,
        val_loader,
        _train_ds,
        _val_ds,
        train_batches,
        _val_batches,
        _n_tokens,
        _n_train_tokens,
    ) = make_loaders(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        path_=data_path,
        split=cfg["train_val_split"],
        shuffle=cfg["shuffle"],
    )

    model = build_model(cfg, tokenizer.vocab_size()).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler, total_steps, _warmup_steps = build_scheduler(
        cfg,
        optimizer,
        max(1, train_batches),
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    resume_path = resolve_resume_path(cfg)
    if resume_path:
        checkpoint = load_ckpt(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed_from={resume_path} start_epoch={start_epoch} global_step={global_step}")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            batch_count += 1

            if global_step >= cfg.get("max_steps", total_steps):
                break

        train_loss = epoch_loss / max(1, batch_count)
        val_loss = evaluate(model, val_loader, device)
        best_val_loss = min(best_val_loss, val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if epoch % cfg["save_every"] == 0:
            save_ckpt(
                weights_path(cfg, epoch),
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
            )

        print(
            f"epoch={epoch} step={global_step} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        if global_step >= cfg.get("max_steps", total_steps):
            break

    sample = sample_text(
        model,
        tokenizer,
        device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return {
        "device": str(device),
        "epochs_ran": len(history),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "history": history,
        "sample": sample,
    }


if __name__ == "__main__":
    train_model()
