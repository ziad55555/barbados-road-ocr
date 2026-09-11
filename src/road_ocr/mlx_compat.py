"""Small, explicit compatibility helpers for MLX-VLM on Apple Silicon."""

from __future__ import annotations


def patch_scalar_repeat() -> None:
    """Accept a scalar MLX array as ``mx.repeat``'s repeat count.

    MLX-VLM 0.5.0's Qwen3-VL vision code passes ``grid_thw[i, 0]`` directly
    to ``mx.repeat``. MLX 0.32 requires a native Python integer. Converting
    only zero-dimensional repeat counts preserves the intended operation and
    avoids modifying installed dependencies.
    """

    import mlx.core as mx

    if getattr(mx.repeat, "_road_scalar_compat", False):
        return

    original_repeat = mx.repeat

    def repeat_compat(array, repeats, axis=None, *, stream=None):
        if isinstance(repeats, mx.array) and repeats.ndim == 0:
            repeats = int(repeats.item())
        kwargs = {} if stream is None else {"stream": stream}
        return original_repeat(array, repeats, axis=axis, **kwargs)

    repeat_compat._road_scalar_compat = True
    mx.repeat = repeat_compat
