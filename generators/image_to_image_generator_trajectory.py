# -*- coding: utf-8 -*-
"""
Image generation generator
"""
from itertools import accumulate
import torch
import math
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from model import LLaDAForMultiModalGeneration

@torch.no_grad()
def generate_image_remask_trajectory(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
    t_window: int = 10,  # === 新增：计算回遮用的最近窗口步数 ===
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    """
    import math
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"
    t_window = timesteps
    # === 新增：按 step 预定义最大回遮数量表 ===
    max_remask_list = torch.zeros(timesteps, dtype=torch.long, device=device)
    for s in range(timesteps):
        r = s / timesteps
        if s == timesteps - 1:
            max_remask_list[s] = 0
        elif 0.4 <= r < 0.6:
            max_remask_list[s] = 16
        elif 0.6 <= r < 0.8:
            max_remask_list[s] = 8
        elif 0.8 <= r < 1:
            max_remask_list[s] = 4
        else:
            max_remask_list[s] = 0

    x = prompt

    vq_mask = x == mask_token_id
    all_should_demask = vq_mask.clone()

    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    if isinstance(model, LLaDAForMultiModalGeneration):
        model.caching(use_cache)
    else:  # DDP
        model.module.caching(use_cache)

    warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool, device=device)
    for step in range(timesteps):
        if not use_cache or step <= warmup_step or (step - warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    # === 新增：为“已去噪 token”维护最近 t_window 步置信度的环形缓冲 ===
    # conf_hist[0, i, j] 存第 i 个位置第 j 槽位的置信度；conf_ptr 指向“下一次写入槽位”索引；conf_cnt 是已写入计数（上限 t_window）
    conf_hist = torch.zeros((B, P, t_window), dtype=torch.float32, device=device)
    conf_ptr  = torch.zeros((B, P), dtype=torch.long, device=device)      # [0, t_window)
    conf_cnt  = torch.zeros((B, P), dtype=torch.int32, device=device)     # [0, t_window]

    # Infer text vocab size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    # 预定义（仅当 use_cache=True 时更新）
    cond_to_compute_mask = None
    uncond_to_compute_mask = None

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_cache and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        already_denoised_mask = all_should_demask & ~vq_mask

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), dim=1)
            uncond_vq_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start-2:]),
                dim=1
            )
            uncond_already_denoised_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size(1)), dtype=torch.bool, device=x.device), already_denoised_mask[:, code_start-2:]),
                dim=1
            )

            cond_logits_full = model(
                x, infer=True, cat='cond', use_cache=use_cache,
                to_compute_mask=cond_to_compute_mask if not refresh_steps[step] else None
            ).logits[..., vocab_offset:vocab_offset + codebook_size]
            cond_mask_logits = cond_logits_full[vq_mask].view(B, -1, codebook_size)
            cond_already_denoised_logits = cond_logits_full[already_denoised_mask].view(B, -1, codebook_size)

            uncond_logits_full = model(
                uncond, infer=True, cat='uncond', use_cache=use_cache,
                to_compute_mask=uncond_to_compute_mask if not refresh_steps[step] else None
            ).logits[..., vocab_offset:vocab_offset + codebook_size]
            uncond_mask_logits = uncond_logits_full[uncond_vq_mask].view(B, -1, codebook_size)
            uncond_already_denoised_logits = uncond_logits_full[uncond_already_denoised_mask].view(B, -1, codebook_size)

            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
            already_denoised_logits = (1 + cfg_scale) * cond_already_denoised_logits - cfg_scale * uncond_already_denoised_logits
        else:
            logits_full = model(x, infer=True).logits[..., vocab_offset:vocab_offset + codebook_size]
            logits = logits_full[:, vq_mask[0], :]
            already_denoised_logits = logits_full[:, already_denoised_mask[0], :]

        # 对尚未确定（mask）的位进行采样
        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [1, N_mask]

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]            # 这些是当前 mask 的位
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        # === 计算“已去噪”的本步置信度，并写入环形缓冲 ===
        num_already_denoised = already_denoised_mask.sum(dim=1, keepdim=True)
        num_remask = 0

        if torch.any(already_denoised_mask):
            already_denoised_sampled = (x[already_denoised_mask] - vocab_offset).unsqueeze(0)  # shape [1, M]
            already_denoised_probs_step = torch.softmax(already_denoised_logits, dim=-1)       # [1, M, K]
            already_denoised_conf_step = already_denoised_probs_step.gather(
                -1, already_denoised_sampled.unsqueeze(-1)
            ).squeeze(-1).to(torch.float32)  # [1, M] -> float32
            print(f"step:{step} already_denoised_conf_step: {already_denoised_conf_step.mean()}")
            # 将本步置信度写入环
            idx1 = already_denoised_mask.nonzero(as_tuple=False)[:, 1]  # [M]
            # 当前这些位置的写指针
            ptr_vals = conf_ptr.view(-1)[idx1]                          # [M]
            # 写入 hist
            # conf_hist.view(-1, t_window)[idx1 * t_window + ptr_vals] = already_denoised_conf_step.view(-1)
            flat_hist = conf_hist.view(-1, t_window)   # [B*P, t_window]，这里 B=1
            rows = idx1                                 # 这些位置对应的“行”
            cols = ptr_vals                              # 每个位置当前应写入的“列”（0..t_window-1）
            flat_hist[rows, cols] = already_denoised_conf_step.view(-1)

            # 更新写指针（mod t_window）
            conf_ptr.view(-1)[idx1] = (ptr_vals + 1) % t_window
            # 更新计数（上限 t_window）
            cnt_vals = conf_cnt.view(-1)[idx1]
            conf_cnt.view(-1)[idx1] = torch.clamp(cnt_vals + 1, max=t_window)

        max_remask = keep_n.new_full(keep_n.size(), max_remask_list[step])
        num_remask = torch.minimum(keep_n, max_remask)

        # === 用“最近 t_window 步平均置信度”做回遮选择 ===
        if torch.all(num_already_denoised > 0) and \
            torch.all(num_remask > 0) and \
                step > 0.4 * timesteps and step < timesteps - 1:

            idx1 = already_denoised_mask.nonzero(as_tuple=False)[:, 1]  # [M]
            # 取出这些位置的历史置信度 [M, t_window] 与有效计数 [M]
            hist_vals = conf_hist.view(-1, t_window)[idx1, :]           # [M, t_window]
            cnt_vals = conf_cnt.view(-1)[idx1].to(torch.float32).clamp_min(1.0)  # [M]
            avg_conf_vals = (hist_vals.sum(dim=-1) / cnt_vals).unsqueeze(0)      # [1, M]
            # 低置信度优先回遮
            # num_remask = torch.min(keep_n, torch.ones_like(keep_n) * 16)

            already_denoised_mask_sel = mask_by_random_topk(
                num_remask.squeeze(1),
                avg_conf_vals,
                temperature=0.0,
                generator=generator
            )
            
            # 执行回遮：把被选中的位置重置为 mask，并清空其历史（以免旧值污染）
            remask_lin = already_denoised_mask_sel.view(-1)
            remask_positions = idx1[remask_lin]
            x.view(-1)[remask_positions] = mask_token_id
            
            avg_remask_conf = avg_conf_vals.view(-1)[already_denoised_mask_sel.view(-1)]
            print(f"step:{step} avg_remask_conf: {avg_remask_conf.mean()}")

            # 清零对应的环与状态
            #conf_hist.view(-1, t_window)[remask_positions, :] = 0
            flat_hist = conf_hist.view(-1, t_window)
            flat_hist[remask_positions, :] = 0
            conf_cnt.view(-1)[remask_positions] = 0
            conf_ptr.view(-1)[remask_positions] = 0

        # 继续正常的（仍为 mask）位置的回遮更新
        mask_sel = mask_by_random_topk((keep_n - num_remask).squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
        print(f"step:{step}, selected token probability: {conf[~mask_sel].mean()}")
        
        # 选择性缓存下一步要算的位置
        if use_cache and step < timesteps - 1 and not refresh_steps[step + 1]:
            if cfg_scale > 0:
                cond_conf = cond_mask_logits.max(dim=-1)[0]
                cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
                cond_to_compute_mask = cond_conf <= cond_conf_threshold

                uncond_conf = uncond_mask_logits.max(dim=-1)[0]
                uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
                uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids
