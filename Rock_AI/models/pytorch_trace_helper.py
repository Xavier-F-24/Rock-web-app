"""Project-owned observable inference instrumentation for dense PyTorch models."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def trace_dense_model(
    model: torch.nn.Module,
    model_inputs: tuple[torch.Tensor, ...],
    *,
    maximum_connections_per_layer: int = 30,
) -> dict[str, Any]:
    """Run one forward pass and retain activations plus strongest local signals."""
    activations: dict[str, float] = {}
    signals: list[dict[str, Any]] = []
    handles = []

    def hook(name: str):
        def capture(module, arguments, output):
            source = arguments[0].detach().reshape(-1)
            result = output.detach().reshape(-1)
            for index, value in enumerate(result):
                activations[f"{name}.node.{index}"] = float(value)
            if not isinstance(module, torch.nn.Linear):
                return
            local = source.unsqueeze(0) * module.weight.detach()
            flat = local.abs().reshape(-1)
            count = min(maximum_connections_per_layer, flat.numel())
            if count <= 0:
                return
            for flat_index in torch.topk(flat, count).indices.tolist():
                target = flat_index // local.shape[1]
                source_index = flat_index % local.shape[1]
                signals.append({
                    "source_id": f"{name}.input.{source_index}",
                    "target_id": f"{name}.node.{target}",
                    "weight": float(module.weight[target, source_index]),
                    "source_activation": float(source[source_index]),
                    "local_signal": float(local[target, source_index]),
                })
        return capture

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            output = model(*model_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "output": float(output.reshape(-1)[0]),
        "node_activations": activations,
        "connection_signals": signals,
    }
