import time
import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 187)) -> float:
    """
    Estimates theoretical Multiply-Accumulate Operations (MACs / FLOPs) for the model.
    Uses thop if available or fallback heuristic calculation.
    """
    try:
        from thop import profile
        dummy = torch.zeros(input_shape)
        device = next(model.parameters()).device
        dummy = dummy.to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs)
    except Exception:
        # Fallback estimation based on parameter operations
        _, total_params = count_parameters(model)
        # Rough heuristic for 1D CNNs: 2 * params * sequence length compression factor
        return float(total_params * input_shape[-1] * 0.5)


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 187),
    device: str = "cpu",
    warmup_runs: int = 20,
    test_runs: int = 100,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Profiles model computational efficiency:
      - Trainable & Total Parameters
      - Model disk size estimate (MB)
      - Single-sample latency (batch=1) in milliseconds (ms)
      - Batched latency (batch=N) in milliseconds (ms)
      - Throughput (samples per second)
      - Peak GPU memory allocation (MB, if CUDA)
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    trainable_p, total_p = count_parameters(model)
    model_size_mb = (total_p * 4) / (1024 * 1024)  # 32-bit float = 4 bytes
    flops = estimate_flops(model, input_shape)

    # 1. Single sample latency (Batch = 1)
    dummy_single = torch.randn(1, *input_shape[1:], device=dev)
    
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_single)
            
        if dev.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        for _ in range(test_runs):
            _ = model(dummy_single)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    single_latency_ms = ((t1 - t0) / test_runs) * 1000.0

    # 2. Batched throughput & latency (Batch = batch_size)
    dummy_batch = torch.randn(batch_size, *input_shape[1:], device=dev)
    
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_batch)
            
        if dev.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        for _ in range(test_runs):
            _ = model(dummy_batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    batched_time_sec = (t1 - t0) / test_runs
    throughput_fps = batch_size / batched_time_sec
    batched_latency_ms = batched_time_sec * 1000.0

    peak_gpu_mem_mb = 0.0
    if dev.type == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    return {
        "trainable_params": int(trainable_p),
        "total_params": int(total_p),
        "model_size_mb": float(model_size_mb),
        "single_latency_ms": float(single_latency_ms),
        "batched_latency_ms": float(batched_latency_ms),
        "throughput_samples_per_sec": float(throughput_fps),
        "peak_gpu_memory_mb": float(peak_gpu_mem_mb),
        "flops": float(flops),
        "device": str(dev)
    }
