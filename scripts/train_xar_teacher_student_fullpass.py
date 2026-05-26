"""Distillation training variant that processes all supervised steps per patch in a single forward pass.

This script keeps the original teacher/student dataset interface but removes KV-cache usage by
building explicit causal masks over the supervised steps within each patch.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Sequence, List
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.trajectory_sequence import TrajectorySequenceDataset
from models import xar
from models.vae import AutoencoderKL
from models.rope import rotate_half

# Reuse utilities from the original training script where possible.
from train_xar_teacher_student import (
    parse_args as base_parse_args,
    setup_distributed,
    select_student_steps,
    parse_step_values,
    prepare_log_labels,
    log_images,
    update_ema,
)


def build_causal_mask(total_segments: int, segment_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a lower-triangular attention mask for a sequence composed of equally sized segments."""
    seq_len = total_segments * segment_len
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    zeros = torch.zeros(1, device=device, dtype=dtype)
    neg_inf = torch.full((1,), float("-inf"), device=device, dtype=dtype)
    mask = torch.where(causal > 0, zeros, neg_inf)
    return mask.unsqueeze(0).unsqueeze(0)


def build_tail_mask(
    history_segments: int,
    num_steps: int,
    segment_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    total_segments = history_segments + num_steps
    if history_segments == 0:
        return build_causal_mask(total_segments, segment_len, device=device, dtype=dtype)

    allowed = torch.zeros(total_segments, total_segments, device=device, dtype=torch.bool)
    for query in range(total_segments):
        for key in range(total_segments):
            if key > query:
                continue
            if key >= history_segments:
                allowed[query, key] = True
                continue
            step_within_patch = key % max(1, num_steps)
            if step_within_patch == num_steps - 1:
                allowed[query, key] = True
    allowed = allowed.repeat_interleave(segment_len, dim=0).repeat_interleave(segment_len, dim=1)
    zeros = torch.zeros(1, device=device, dtype=dtype)
    neg_inf = torch.full((1,), float("-inf"), device=device, dtype=dtype)
    mask = torch.where(allowed, zeros, neg_inf)
    return mask.unsqueeze(0).unsqueeze(0)


def patch_rope_for_index_map(rope_module):
    if getattr(rope_module, "_index_map_ready", False):
        return

    original_forward = rope_module.forward

    def forward_with_index(self, x, scale_index=None):
        index_map = getattr(self, "_index_map", None)
        if index_map is not None:
            if index_map.shape[0] != x.shape[2]:
                idx = index_map[: x.shape[2]]
            else:
                idx = index_map
            freqs_cos = self.freqs_cos[idx].unsqueeze(0).unsqueeze(0)
            freqs_sin = self.freqs_sin[idx].unsqueeze(0).unsqueeze(0)
            return x * freqs_cos + rotate_half(x) * freqs_sin
        return original_forward(x, scale_index)

    rope_module.forward = types.MethodType(forward_with_index, rope_module)
    rope_module._index_map_ready = True


def set_rope_index_map(blocks, index_map: torch.Tensor):
    for blk in blocks:
        rope = blk.attn.rope
        patch_rope_for_index_map(rope)
        rope._index_map = index_map


def clear_rope_index_map(blocks):
    for blk in blocks:
        rope = blk.attn.rope
        if hasattr(rope, "_index_map"):
            rope._index_map = None


def forward_patch_trajectory(
    base_model: torch.nn.Module,
    patch_tokens: torch.Tensor,
    label: torch.Tensor,
    patch_idx: int,
    timesteps: torch.Tensor,
    cfg_scale: float,
    *,
    history_tokens: torch.Tensor | None,
    history_timesteps: torch.Tensor | None,
    retain_encoder_layers: int,
    retain_decoder_layers: int,
) -> torch.Tensor:
    """Run the student once per patch over all supervised steps using a causal mask.

    Args:
        base_model: The underlying xAR model (not wrapped by DDP).
        patch_tokens: Tensor[B, S, L, C] with teacher latents for the supervised patch.
        label: Tensor[B] class labels.
        patch_idx: Which patch (0-based) is currently supervised.
        timesteps: Tensor[B, S] of diffusion timesteps per supervised step in this patch.
        cfg_scale: Classifier-free guidance scale.
        history_tokens: Optional Tensor[B, H, L, C] containing all previous patches' supervised
            steps flattened into `H` segments (patch order followed by step order).
        history_timesteps: Optional Tensor[B, H] with the corresponding diffusion timesteps.

    Returns:
        Tensor[B, S, L, C] of student predictions aligned with `patch_tokens` order.
    """
    device = patch_tokens.device
    dtype = patch_tokens.dtype
    batch_size, num_steps, segment_len, embed_dim = patch_tokens.shape

    segments: List[torch.Tensor] = []
    time_vectors: List[torch.Tensor] = []

    history_segment_count = 0
    if history_tokens is not None and history_timesteps is not None:
        history_tokens = history_tokens.to(device=device, dtype=dtype)
        history_timesteps = history_timesteps.to(device=device, dtype=dtype)
        history_segment_count = history_tokens.shape[1]
        for seg_idx in range(history_segment_count):
            segments.append(history_tokens[:, seg_idx])
            time_vectors.append(history_timesteps[:, seg_idx])

    for step_id in range(num_steps):
        segments.append(patch_tokens[:, step_id].to(device=device, dtype=dtype))
        time_vectors.append(timesteps[:, step_id].to(device=device, dtype=dtype))

    total_segments = len(segments)
    sequence_tokens = torch.cat(segments, dim=1)  # [B, total_segments * L, C]
    base_batch = sequence_tokens.shape[0]

    time_values = torch.cat([vec.reshape(-1) for vec in time_vectors], dim=0)  # [total_segments * B]

    batch_tokens = sequence_tokens
    labels = label
    cfg_active = cfg_scale > 1.0
    if cfg_active:
        batch_tokens = torch.cat([batch_tokens, batch_tokens], dim=0)
        labels = torch.cat([label, torch.full_like(label, 1000)], dim=0)
        time_values = torch.cat([time_values, time_values], dim=0)

    # Embed tokens and conditions.
    token_embeddings = base_model.z_proj(batch_tokens)

    scaled_timesteps = (time_values * 1000).long()
    time_condition = base_model.time_embed(scaled_timesteps).squeeze()
    class_condition = base_model.class_emb(labels.long()).squeeze()

    if class_condition.ndim == 1:
        class_condition = class_condition.unsqueeze(0)
    if time_condition.ndim == 1:
        time_condition = time_condition.unsqueeze(0)

    step_factor = total_segments
    condition = class_condition.repeat(step_factor, 1) + time_condition

    causal_mask = build_causal_mask(
        total_segments=total_segments,
        segment_len=segment_len,
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )
    tail_mask = build_tail_mask(
        history_segment_count,
        num_steps,
        segment_len,
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )

    encoder_full_limit = len(base_model.encoder_blocks) if retain_encoder_layers < 0 else min(retain_encoder_layers, len(base_model.encoder_blocks))
    decoder_full_limit = len(base_model.decoder_blocks) if retain_decoder_layers < 0 else min(retain_decoder_layers, len(base_model.decoder_blocks))

    owner_indices: List[int] = []
    for seg_idx in range(history_segment_count):
        owner_indices.append(seg_idx // max(1, num_steps))
    owner_indices.extend([patch_idx] * num_steps)

    def make_positional_embedding(raw_embed: torch.Tensor) -> torch.Tensor:
        segments: List[torch.Tensor] = []
        for owner in owner_indices:
            start = owner * segment_len
            end = start + segment_len
            segments.append(raw_embed[:, start:end])
        return torch.cat(segments, dim=1).to(device=token_embeddings.device, dtype=token_embeddings.dtype)

    encoder_pos_embed = make_positional_embedding(base_model.encoder_pos_embed_learned)
    decoder_pos_embed = make_positional_embedding(base_model.decoder_pos_embed_learned)

    index_map = torch.tensor(
        [owner * segment_len + offset for owner in owner_indices for offset in range(segment_len)],
        device=token_embeddings.device,
        dtype=torch.long,
    )

    def encoder_forward(x_tokens: torch.Tensor) -> torch.Tensor:
        set_rope_index_map(base_model.encoder_blocks, index_map)
        x = x_tokens + encoder_pos_embed
        x = base_model.z_proj_ln(x)
        for idx, blk in enumerate(base_model.encoder_blocks):
            mask = causal_mask if idx < encoder_full_limit else tail_mask
            x = blk(x, condition, mask, update_cache=False, scale_index=patch_idx)
        x = base_model.encoder_norm(x)
        clear_rope_index_map(base_model.encoder_blocks)
        return x

    def decoder_forward(x_tokens: torch.Tensor) -> torch.Tensor:
        set_rope_index_map(base_model.decoder_blocks, index_map)
        x = base_model.decoder_embed(x_tokens)
        x = x + decoder_pos_embed
        for idx, blk in enumerate(base_model.decoder_blocks):
            mask = causal_mask if idx < decoder_full_limit else tail_mask
            x = blk(x, condition, mask, update_cache=False, scale_index=patch_idx)
        x = base_model.decoder_norm(x)
        x = base_model.pred(x)
        clear_rope_index_map(base_model.decoder_blocks)
        return x

    encoded = encoder_forward(token_embeddings)
    decoded = decoder_forward(encoded)

    if cfg_active:
        cond, uncond = decoded.chunk(2, dim=0)
        decoded = uncond + cfg_scale * (cond - uncond)

    context_segments = history_segment_count
    start = context_segments * segment_len
    useful = decoded[:base_batch, start:]
    preds = useful.reshape(batch_size, num_steps, segment_len, -1)
    return preds


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg_scale: float,
    epoch: int,
    *,
    selected_indices: Sequence[int],
    writer: SummaryWriter | None,
    rank: int,
    steps_per_epoch: int,
    total_epochs: int,
    image_interval: int,
    image_labels: Sequence[int],
    image_steps: Sequence[int],
    vae: AutoencoderKL | None,
    image_seed: int,
    grad_accum_steps: int,
    clip_grad_norm: float,
    log_interval: int,
    ema_model: torch.nn.Module | None,
    ema_decay: float,
    global_start_time: float,
    retain_encoder_layers: int,
    retain_decoder_layers: int,
) -> float:
    base_model = model.module if isinstance(model, DDP) else model
    base_model.train()

    grad_accum_steps = max(1, grad_accum_steps)
    total_loss = 0.0
    total_batches = 0
    total_training_steps = steps_per_epoch * total_epochs
    interval_loss = 0.0
    interval_batches = 0
    interval_start = time.time()

    num_patches = base_model.clusters
    num_steps = len(selected_indices)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        xt = batch["xt"].to(device)
        velocity = batch["velocity"].to(device)
        timestep = batch["timestep"].to(device)
        label = batch["label"].to(device)

        global_step = epoch * steps_per_epoch + batch_idx
        patch_idx = global_step % num_patches

        patch_xt = xt[:, patch_idx, selected_indices]
        patch_timesteps = timestep[:, patch_idx, selected_indices]

        history_tokens = None
        history_timesteps = None
        if patch_idx > 0:
            history_tokens = xt[:, :patch_idx, selected_indices]
            history_tokens = history_tokens.contiguous().view(
                xt.shape[0], -1, patch_xt.shape[2], patch_xt.shape[3]
            )
            history_timesteps = timestep[:, :patch_idx, selected_indices]
            history_timesteps = history_timesteps.contiguous().view(
                timestep.shape[0], -1
            )

        preds = forward_patch_trajectory(
            base_model,
            patch_xt,
            label,
            patch_idx,
            patch_timesteps,
            cfg_scale,
            history_tokens=history_tokens,
            history_timesteps=history_timesteps,
            retain_encoder_layers=retain_encoder_layers,
            retain_decoder_layers=retain_decoder_layers,
        )

        target = velocity[:, patch_idx, selected_indices]
        loss = F.mse_loss(preds, target)

        (loss / grad_accum_steps).backward()
        batch_loss = loss.item()

        total_loss += batch_loss
        total_batches += 1
        interval_loss += batch_loss
        interval_batches += 1

        global_step = epoch * steps_per_epoch + batch_idx
        if rank == 0:
            elapsed_global = max(time.time() - global_start_time, 1e-6)
            steps_done = global_step + 1
            steps_per_sec_global = steps_done / elapsed_global
            remaining_steps = max(total_training_steps - steps_done, 0)
            eta_seconds = remaining_steps / max(steps_per_sec_global, 1e-6)
            elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_global))
            eta_hms = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            print(
                f"[Progress] step {steps_done}/{total_training_steps} "
                f"(epoch {epoch + 1}/{total_epochs}, patch {patch_idx}) "
                f"batch_loss={batch_loss:.6f} elapsed={elapsed_hms} eta={eta_hms} "
                f"speed={steps_per_sec_global:.2f} it/s"
            )

        if writer is not None and rank == 0:
            writer.add_scalar("train/loss_batch", batch_loss, global_step)
            if image_interval > 0 and global_step > 0 and (global_step % image_interval == 0):
                log_images(
                    writer,
                    ema_model if ema_model is not None else model,
                    vae,
                    labels=image_labels,
                    step_values=image_steps,
                    cfg_scale=cfg_scale,
                    device=device,
                    seed=image_seed,
                    global_step=global_step,
                )

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == steps_per_epoch - 1)
        if should_step:
            if clip_grad_norm > 0:
                clip_grad_norm_(base_model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                update_ema(ema_model, base_model, ema_decay)

        if (
            rank == 0
            and log_interval > 0
            and (global_step + 1) % log_interval == 0
        ):
            elapsed = max(time.time() - interval_start, 1e-6)
            steps_per_sec = interval_batches / elapsed
            avg_interval_loss = interval_loss / max(1, interval_batches)
            print(
                f"[Stats] step={global_step + 1:06d} loss={avg_interval_loss:.6f} "
                f"({steps_per_sec:.2f} it/s)"
            )
            if writer is not None:
                writer.add_scalar("train/loss_interval", avg_interval_loss, global_step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)
            interval_loss = 0.0
            interval_batches = 0
            interval_start = time.time()

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def main() -> None:
    args = base_parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)

    dataset = TrajectorySequenceDataset(args.trajectory_dir, max_samples=args.max_samples)
    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    explicit_values = None
    if args.student_step_values:
        explicit_values = [int(token.strip()) for token in args.student_step_values.split(",") if token.strip()]

    selected_indices, selected_steps = select_student_steps(
        dataset.step_ids,
        explicit_values=explicit_values,
        target_count=args.student_num_steps,
    )

    if rank == 0:
        print(f"Using student supervision steps: {selected_steps}")

    image_step_values = parse_step_values(args.image_step_values) if args.image_step_values else []
    if rank == 0 and not image_step_values and args.image_log_interval > 0:
        print("[WARN] image-step-values is empty; disabling image logging")
    image_labels_arg = parse_step_values(args.image_log_labels) if args.image_log_labels else []
    log_labels = prepare_log_labels(dataset, image_labels_arg, args.image_log_count)

    constructor = {
        "base": xar.xar_base,
        "large": xar.xar_large,
        "huge": xar.xar_huge,
    }[args.model_size]

    model = constructor(
        img_size=256,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.requires_grad_(True)
    model.train()

    ema_model: torch.nn.Module | None = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    base_model = model.module if isinstance(model, DDP) else model
    if ema_model is not None:
        update_ema(ema_model, base_model, 0.0)
    log_model_for_images = ema_model if ema_model is not None else base_model

    vae: AutoencoderKL | None = None
    if (
        args.image_log_interval > 0
        and args.vae_path
        and rank == 0
    ):
        vae = AutoencoderKL(
            embed_dim=16,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=args.vae_path,
        ).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
    elif rank == 0 and args.image_log_interval > 0:
        print("[WARN] image logging requested but --vae-path not provided; disabling image logs")
        image_step_values = []
        log_labels = []

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 72)
        print("Teacher-architecture student distillation (full-pass variant)")
        print(f"World size    : {world_size}")
        print(f"Output dir    : {output_dir}")
        print(f"Log dir       : {Path(args.log_dir) if args.log_dir else Path('distill_tensorboard_output') / output_dir.name}")
        print(f"LR            : {args.lr}")
        print(f"CFG scale     : {args.cfg}")
        print(f"Step schedule : {selected_steps}")
        print(f"Grad accum    : {args.grad_accumulation_steps}")
        print(f"Clip grad norm: {args.clip_grad_norm}")
        print(f"Log interval  : {args.log_interval}")
        print(f"EMA decay     : {args.ema_decay}")
        if args.image_log_interval > 0 and image_step_values:
            print(f"Image steps   : {image_step_values}")
            print(f"Image labels  : {log_labels}")
        else:
            print("Image logging : disabled")
        print(f"Dataset size  : {len(dataset)}")
        print(f"Batches/epoch : {len(loader)}")
        print("=" * 72)

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_root = Path("distill_tensorboard_output")
        if rank == 0:
            log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / output_dir.name
    writer = SummaryWriter(log_dir=str(log_dir)) if rank == 0 else None

    if writer is not None and vae is not None and args.image_log_interval > 0 and image_step_values:
        log_images(
            writer,
            log_model_for_images,
            vae,
            labels=log_labels,
            step_values=image_step_values,
            cfg_scale=args.cfg,
            device=device,
            seed=args.image_log_seed,
            global_step=0,
        )
        if rank == 0:
            print(f"[INFO] Initial student samples logged to {log_dir} (global_step=0)")

    steps_per_epoch = len(loader)
    total_epochs = args.epochs
    vae_for_rank = vae if rank == 0 else None
    global_start_time = time.time()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            args.cfg,
            epoch,
            selected_indices=selected_indices,
            writer=writer,
            rank=rank,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            image_interval=args.image_log_interval,
            image_labels=log_labels,
            image_steps=image_step_values,
            vae=vae_for_rank,
            image_seed=args.image_log_seed,
            grad_accum_steps=args.grad_accumulation_steps,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            global_start_time=global_start_time,
            retain_encoder_layers=args.retain_encoder_layers,
            retain_decoder_layers=args.retain_decoder_layers,
        )
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}")
            checkpoint = {
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema_model.state_dict() if ema_model is not None else None,
                "args": vars(args),
            }
            torch.save(
                checkpoint,
                output_dir / f"student_teacher_fullpass_epoch{epoch+1}.pth",
            )

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
