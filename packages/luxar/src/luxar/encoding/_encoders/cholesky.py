"""Cholesky split encoder: the owned diag/offdiag precision policy, the
encode-time covariance certificate, and the plain CHOLESKY dtype encoder."""

import warnings
from typing import Any, Optional

import numpy as np
import zarr

from ..compression import resolve_compressor
from ..modes import EncodingMode
from ..semantic_types import SemanticType
from .base import BaseEncoderMixin

# Escalation threshold for the AUTO covariance certificate: p95 over splats of
# the relative Frobenius error of Σ = L·Lᵀ after a quantized round-trip. AUTO
# tries u8 first and only escalates (u16, then float32) when the measured error
# exceeds this bound — so it is a hard invariant of AUTO output, not a heuristic.
# Calibration (2026-07 covariance spike, real light-sheet fit): u8 measured
# relF p95 ≈ 0.02 while rendering ~46 dB below the fit-error floor (invisible);
# 0.05 keeps a wide safety margin yet is reachable by genuinely hard data
# (e.g. merged stores whose σ columns span many decades).
COV_CERT_RELF_P95_MAX = 0.05

#: Row cap for the certificate measurement. The p95 statistic needs a bounded
#: sample, not every splat — without a cap, a 10M-splat flat fit would build
#: ~2.7 GB of transient float64 Σ scratch just to certify. Evenly-spaced rows
#: keep the sample deterministic and (over Hilbert-ordered splats) spatially
#: uniform; the quantization scales always come from the full columns.
COV_CERT_SAMPLE_MAX = 262_144


class CholeskyEncoderMixin(BaseEncoderMixin):
    """Cholesky split-pair encoding policy for :class:`ArrayEncoder`."""

    def encode_cholesky_split(
        self,
        zarr_group: zarr.Group,
        diag: np.ndarray,
        offdiag: np.ndarray,
        ndim: int,
        mode: EncodingMode = EncodingMode.AUTO,
        *,
        diag_name: str = "cholesky_factors_diag",
        offdiag_name: str = "cholesky_factors_offdiag",
        n_elements: Optional[int] = None,
        chunks_diag: Optional[tuple] = None,
        chunks_offdiag: Optional[tuple] = None,
        compressor: Optional[Any] = None,
        certificate_threshold: float = COV_CERT_RELF_P95_MAX,
    ) -> None:
        """Encode a split Cholesky pair (diag + offdiag) under one owned policy.

        The joint entry point for CHOLESKY_DIAG/CHOLESKY_OFFDIAG pairs: both
        halves always land on the SAME precision tier, and the tier choice is
        made here — callers never pass bit widths or run quality checks.

        Policy: PRECISION → float32; MEMORY → u8 unconditionally; AUTO → u8
        with an encode-time **certificate** — the pair is round-tripped through
        the exact quantization transform, Σ = L·Lᵀ is rebuilt from both halves,
        and the p95 per-splat relative Frobenius error is measured. If it
        exceeds ``certificate_threshold`` (default
        :data:`COV_CERT_RELF_P95_MAX`), AUTO escalates to u16, and — should
        even u16 fail (practically unreachable) — to float32. The measured
        certificate is recorded in each array's own ``encoding`` attrs as pure
        provenance (never needed for decode; arrays stay self-describing).

        1D gsplats have no off-diagonal terms: an ``offdiag`` with zero columns
        is accepted and simply not written (readers reconstruct the empty
        block). Both arrays are written with ``deduplicate=False`` — the
        per-channel scales live in each array's own attrs, so an ``array_ref``
        would strand the viewer without them.
        """
        diag = np.asarray(diag)
        offdiag = np.asarray(offdiag)
        n_off = ndim * (ndim - 1) // 2
        if diag.ndim != 2 or diag.shape[1] != ndim:
            raise ValueError(
                f"diag must have shape (N, {ndim}) for ndim={ndim}; got {diag.shape}"
            )
        if offdiag.ndim != 2 or offdiag.shape[1] != n_off:
            raise ValueError(
                f"offdiag must have shape (N, {n_off}) for ndim={ndim}; "
                f"got {offdiag.shape}"
            )
        write_offdiag = offdiag.shape[1] > 0

        # AUTO: certify u8, escalate on measured Σ error. Skipped for a uniform
        # pair — the broadcast priority path stores the exact float row, so
        # there is no quantization to certify. The n_rows > 0 short-circuit
        # also protects _is_uniform (which indexes data[0]) from empty arrays:
        # zero-splat pairs fall straight through to encode()'s size==0
        # passthrough, matching the pre-joint-call behavior.
        n_rows = diag.shape[0]
        pair_uniform = n_rows > 0 and (
            self._is_uniform(diag) and (not write_offdiag or self._is_uniform(offdiag))
        )
        certificate: Optional[dict] = None
        chosen_bits: Optional[int] = 8
        eff_mode = mode
        if mode == EncodingMode.AUTO and n_rows > 0 and not pair_uniform:
            # The percentile only needs a bounded SAMPLE of rows, so the f64 Σ
            # scratch stays capped (a 10M-splat flat fit would otherwise build
            # ~2.7 GB of transient Σ arrays). Evenly-spaced rows = deterministic
            # and, over Hilbert-ordered splats, spatially uniform. The
            # quantization SCALES, however, must come from the FULL columns —
            # exactly what the real encode uses — or the certificate lies;
            # _perchannel_log_scales is the shared nonzero-anchored reduction
            # both the encoders and this certificate use.
            capped = n_rows > COV_CERT_SAMPLE_MAX
            sel: Any = (
                np.linspace(0, n_rows - 1, COV_CERT_SAMPLE_MAX).astype(np.intp)
                if capped
                else slice(None)
            )
            diag_s = diag[sel]
            off_s = offdiag[sel]
            n_s = diag_s.shape[0]

            lo_d, hi_d = self._perchannel_log_scales(diag, signed=False)
            lo_o, hi_o = (
                self._perchannel_log_scales(offdiag, signed=True)
                if write_offdiag
                else (None, None)
            )

            # Reference for the error measurement is the FORWARD-VALID input:
            # the log encoder clamps invalid negative diagonal entries by
            # policy, and that validation loss must not read as quantization
            # error (it is identical at every tier, so escalating cannot
            # recover it). The reference Σ is tier-independent — build it once,
            # outside the escalation loop.
            diag_ref = np.maximum(diag_s.astype(np.float64), 0.0)
            s_ref = self._sigma_from_split(diag_ref, off_s, ndim).reshape(n_s, -1)
            den = np.maximum(np.linalg.norm(s_ref, axis=1), 1e-30)
            tried: list[tuple[int, float]] = []
            chosen_bits = None
            for bits in (8, 16):
                diag_q = self._perchannel_log_roundtrip(
                    diag_s, bits, signed=False, lo=lo_d, hi=hi_d
                )
                off_q = (
                    self._perchannel_log_roundtrip(
                        off_s, bits, signed=True, lo=lo_o, hi=hi_o
                    )
                    if write_offdiag
                    else off_s
                )
                s_q = self._sigma_from_split(diag_q, off_q, ndim).reshape(n_s, -1)
                relf = self._relf_p95(s_ref, den, s_q)
                tried.append((bits, relf))
                if relf <= certificate_threshold:
                    chosen_bits = bits
                    break
            if chosen_bits is not None:
                tier = f"u{chosen_bits}"
                value = tried[-1][1]
            else:
                tier, value = "float32", 0.0
                eff_mode = EncodingMode.PRECISION
            certificate = {
                "metric": "cov_relf_p95",
                "value": float(value),
                "threshold": float(certificate_threshold),
                "tier": tier,
            }
            if capped:
                certificate["sample"] = int(n_s)
            if tier == "u16":
                warnings.warn(
                    f"CHOLESKY '{diag_name}': covariance certificate "
                    f"cov_relf_p95 = {tried[0][1]:.4g} > {certificate_threshold} "
                    f"at uint8 — escalating to uint16 "
                    f"(measured {tried[1][1]:.4g}).",
                    UserWarning,
                    stacklevel=2,
                )
            elif tier == "float32":
                warnings.warn(
                    f"CHOLESKY '{diag_name}': covariance certificate "
                    f"cov_relf_p95 = {tried[1][1]:.4g} > {certificate_threshold} "
                    "even at uint16 — storing float32.",
                    UserWarning,
                    stacklevel=2,
                )

        # Delegate each half to the full encode() priority ladder (broadcast /
        # LUT / dtype) with the certified tier forced for the quantized path.
        # The tier is threaded explicitly via `_perchannel_bits` (consumed only
        # by the CHOLESKY_DIAG/OFFDIAG per-channel log encoders) — no shared
        # instance state, so this is re-entrant and needs no try/finally.
        self.encode(
            data=diag,
            zarr_group=zarr_group,
            name=diag_name,
            semantic_type=SemanticType.CHOLESKY_DIAG,
            mode=eff_mode,
            n_elements=n_elements,
            chunks=chunks_diag,
            compressor=compressor,
            deduplicate=False,
            _perchannel_bits=chosen_bits,
        )
        if write_offdiag:
            self.encode(
                data=offdiag,
                zarr_group=zarr_group,
                name=offdiag_name,
                semantic_type=SemanticType.CHOLESKY_OFFDIAG,
                mode=eff_mode,
                n_elements=n_elements,
                chunks=chunks_offdiag,
                compressor=compressor,
                deduplicate=False,
                _perchannel_bits=chosen_bits,
            )

        # Record the certificate as provenance inside each array's own encoding
        # attrs — only where the tier decision actually applied (the broadcast /
        # LUT priority paths store exact values, so no certificate there).
        if certificate is not None:
            targets = [diag_name] + ([offdiag_name] if write_offdiag else [])
            for arr_name in targets:
                if arr_name not in zarr_group:
                    continue
                enc = dict(zarr_group[arr_name].attrs.get("encoding", {}))
                enc_name = str(enc.get("name", ""))
                quantized = enc_name.startswith(
                    ("log_perchannel", "signed_log_perchannel")
                )
                if quantized or (
                    certificate["tier"] == "float32" and enc_name == "float32"
                ):
                    enc["certificate"] = certificate
                    zarr_group[arr_name].attrs["encoding"] = enc

    @staticmethod
    def _cov_relf_p95(
        diag: np.ndarray,
        diag_q: np.ndarray,
        offdiag: np.ndarray,
        offdiag_q: np.ndarray,
        ndim: int,
    ) -> float:
        """p95 over splats of the relative Frobenius error of Σ = L·Lᵀ.

        Composed from :meth:`_sigma_from_split` + :meth:`_relf_p95`; the
        escalation loop in :meth:`encode_cholesky_split` uses the pieces
        directly so the tier-independent reference Σ is built only once.
        """
        n = diag.shape[0]
        s0 = CholeskyEncoderMixin._sigma_from_split(diag, offdiag, ndim).reshape(n, -1)
        sq = CholeskyEncoderMixin._sigma_from_split(diag_q, offdiag_q, ndim).reshape(n, -1)
        den = np.maximum(np.linalg.norm(s0, axis=1), 1e-30)
        return CholeskyEncoderMixin._relf_p95(s0, den, sq)

    @staticmethod
    def _sigma_from_split(
        diag: np.ndarray, offdiag: np.ndarray, ndim: int
    ) -> np.ndarray:
        """(N, d, d) Σ = L·Lᵀ from split Cholesky halves.

        Rebuilds lower-triangular L with a local row-major
        ``np.tril_indices`` layout — the same packing convention as
        ``gsplats.utils.trils`` (locked by a parity test), kept local so the
        encoding package takes no gsplats dependency.
        """
        n = diag.shape[0]
        rows, cols = np.tril_indices(ndim)
        off = rows != cols
        tri = np.zeros((n, ndim, ndim), dtype=np.float64)
        tri[:, np.arange(ndim), np.arange(ndim)] = diag.astype(np.float64)
        if offdiag.shape[1] > 0:
            tri[:, rows[off], cols[off]] = offdiag.astype(np.float64)
        return np.asarray(np.einsum("nij,nkj->nik", tri, tri))

    @staticmethod
    def _relf_p95(s_ref: np.ndarray, den: np.ndarray, s_q: np.ndarray) -> float:
        """p95 of per-row relative Frobenius error given flattened Σ matrices."""
        num = np.linalg.norm(s_q - s_ref, axis=1)
        return float(np.percentile(num / den, 95))

    def _encode_cholesky(
        self,
        zarr_group: zarr.Group,
        name: str,
        data: np.ndarray,
        mode: EncodingMode,
        chunks: Optional[tuple] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Encode CHOLESKY semantic type.

        Args:
            zarr_group: Zarr group to write to
            name: Array name
            data: Array data
            mode: Encoding mode
            chunks: Optional chunk shape
            compressor: Optional compressor
        """
        original_dtype = str(data.dtype)

        target_dtype: np.dtype[Any]
        if mode == EncodingMode.PRECISION or mode == EncodingMode.AUTO:
            target_dtype = np.dtype("float32")
        elif mode == EncodingMode.MEMORY:
            # Check if float16 is allowed, fallback to float32 if not
            if self._float16_allowed:
                target_dtype = np.dtype("float16")
            else:
                target_dtype = np.dtype("float32")
        else:
            raise ValueError(f"Unexpected mode for CHOLESKY: {mode}")

        encoded_data = data.astype(target_dtype)
        zarr_group.create_dataset(
            name,
            data=encoded_data,
            chunks=chunks,
            compressor=resolve_compressor(compressor, encoded_data.dtype),
            overwrite=True,
        )
        zarr_group[name].attrs["encoding"] = {
            "name": target_dtype.name,
            "original_dtype": original_dtype,
        }

