"""SVMD decomposition utilities with entropy-based mode selection.

This module wraps a lightweight SVMD-inspired workflow that can be reused by
multiple datasets. It relies on ``entropyhub`` for the ``SlopEn`` metric and
PyWavelets for the actual signal splitting; both dependencies are available via
``environment.yml``.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pywt

try:
    import entropyhub as eh
except ImportError as exc:  # pragma: no cover - handled by dependency installation
    raise ImportError(
        "entropyhub is required for SVMD utilities. Install it with `pip install entropyhub` "
        "or add it to your environment file."
    ) from exc


@dataclass
class SVMDSettings:
    """Configuration for SVMD processing."""

    use_svmd: bool = False
    k: int = 5
    max_iter: int = 500
    max_effective_modes: int = 10
    max_runtime: float = 5.0
    downsample_stride: int = 1
    long_series_length: int = 5000
    long_series_max_iter: int = 250
    cache_dir: Optional[str] = None
    slopen_m: int = 3
    slopen_tau: int = 1
    wavelet: str = "db4"
    min_points_per_mode: int = 32


class SVMD:
    """A robust, entropy-guided SVMD helper.

    The implementation uses wavelet decomposition as the backbone for mode
    extraction and leverages Slope Entropy (SlopEn) to prioritise informative
    modes. Runtime and parameter guards make the process resilient for long
    sequences.
    """

    def __init__(self, settings: SVMDSettings) -> None:
        self.settings = settings

    def _adjust_parameters(self, length: int) -> tuple[int, int, int, int]:
        adaptive_k = max(1, min(self.settings.k, length // self.settings.min_points_per_mode))
        adaptive_max_modes = max(1, min(self.settings.max_effective_modes, adaptive_k))
        adaptive_iter = min(
            self.settings.max_iter,
            self.settings.long_series_max_iter if length > self.settings.long_series_length else self.settings.max_iter,
        )
        stride = max(1, self.settings.downsample_stride)
        return adaptive_k, adaptive_max_modes, adaptive_iter, stride

    def _decompose_signal(self, signal: np.ndarray, level: int) -> list[np.ndarray]:
        coeffs = pywt.wavedec(signal, self.settings.wavelet, level=level)
        modes: list[np.ndarray] = []

        for idx in range(len(coeffs)):
            isolated = [np.zeros_like(arr) for arr in coeffs]
            isolated[idx] = coeffs[idx]
            mode = pywt.waverec(isolated, self.settings.wavelet)[: signal.shape[0]]
            modes.append(mode)
        return modes

    def _rank_modes(self, modes: Iterable[np.ndarray]) -> list[np.ndarray]:
        scored_modes = []
        for mode in modes:
            energy = float(np.linalg.norm(mode))
            try:
                slopen_value = float(eh.SlopEn(mode, m=self.settings.slopen_m, tau=self.settings.slopen_tau)[0])
            except Exception:
                slopen_value = 0.0
            score = energy * (1.0 + slopen_value)
            scored_modes.append((score, mode))

        scored_modes.sort(key=lambda item: item[0], reverse=True)
        return [mode for _, mode in scored_modes]

    def decompose(self, signal: np.ndarray) -> list[np.ndarray]:
        start_time = time.time()
        signal = np.asarray(signal, dtype=float).flatten()
        length = signal.shape[0]

        adaptive_k, adaptive_max_modes, adaptive_iter, stride = self._adjust_parameters(length)
        working_signal = signal
        if stride > 1 and length // stride > adaptive_k:
            indices = np.arange(0, length, stride)
            working_signal = np.interp(indices, np.arange(length), signal)

        max_level = pywt.dwt_max_level(data_len=working_signal.shape[0], filter_len=pywt.Wavelet(self.settings.wavelet).dec_len)
        level = max(1, min(adaptive_k, adaptive_iter, max_level))

        raw_modes = self._decompose_signal(working_signal, level=level)
        ranked_modes = self._rank_modes(raw_modes)

        selected_modes: list[np.ndarray] = []
        for mode in ranked_modes:
            if time.time() - start_time > self.settings.max_runtime:
                break
            selected_modes.append(mode)
            if len(selected_modes) >= adaptive_max_modes:
                break

        if working_signal is not signal:
            full_indices = np.arange(length)
            selected_modes = [np.interp(full_indices, np.arange(0, length, stride)[: mode.shape[0]], mode) for mode in selected_modes]

        return selected_modes

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        modes = self.decompose(signal)
        if not modes:
            return np.asarray(signal, dtype=float)
        return np.sum(modes, axis=0)

    def batch_denoise(self, matrix: np.ndarray) -> np.ndarray:
        processed = np.empty_like(matrix, dtype=float)
        for col in range(matrix.shape[1]):
            processed[:, col] = self.denoise(matrix[:, col])
        return processed


def _build_cache_path(cache_root: str, cache_key: str, settings: SVMDSettings) -> str:
    os.makedirs(cache_root, exist_ok=True)
    digest_source = f"{cache_key}_{settings.k}_{settings.max_effective_modes}_{settings.downsample_stride}"
    digest = hashlib.md5(digest_source.encode()).hexdigest()[:10]
    return os.path.join(cache_root, f"svmd_{digest}.npy")


def apply_svmd_with_cache(
    data: np.ndarray, *, cache_root: str, cache_key: str, settings: SVMDSettings
) -> np.ndarray:
    if not settings.use_svmd:
        return data

    cache_path = _build_cache_path(cache_root, cache_key, settings)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    svmd = SVMD(settings)
    processed = svmd.batch_denoise(data)
    np.save(cache_path, processed)
    return processed
