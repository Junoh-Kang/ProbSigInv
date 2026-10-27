"""Augment paths and compute (log-)signatures via iisignature."""

from math import factorial

import numpy as np
import iisignature


def augment_path(log_prices: np.ndarray, time_grid: np.ndarray, aug_type: str = "TA") -> np.ndarray:
    """Prepend time channel(s) to log-price paths.

    Parameters
    ----------
    log_prices : (P, T+1) or (P, T+1, D) array of log(S_t / S0)
    time_grid  : (T+1,) normalized time in [0, 1]
    aug_type   : "LS"/"TA" (default) → [t, X(t)],                shape (P, T+1,   2)
                 "fourier"               → [t, sin(t), cos(t)-1, X(t)], shape (P, T+1, 4)
                 "TLL"                   → lead-lag of [t, X(t)],  shape (P, 2T+1,  4)
                 "TLL3"                  → lead-lag without t^lag, shape (P, 2T+1,  3)

    Returns
    -------
    augmented : (P, T+1, C) or (P, 2T+1, C) for lead-lag variants
    """
    log_prices = np.asarray(log_prices)
    time_grid = np.asarray(time_grid)
    if log_prices.ndim == 2:
        path_values = log_prices[:, :, None]
    elif log_prices.ndim == 3:
        path_values = log_prices
    else:
        raise ValueError(
            f"log_prices must have shape (P, T) or (P, T, D), got {log_prices.shape}"
        )

    P, L, D = path_values.shape
    if time_grid.ndim != 1 or time_grid.shape[0] != L:
        raise ValueError(
            f"time_grid must have shape ({L},) for paths shape {log_prices.shape}, got {time_grid.shape}"
        )

    time_col = np.broadcast_to(time_grid[None, :], (P, L))  # (P, T+1)
    time_channel = time_col[:, :, None]
    if aug_type in ("LS", "TA"):
        augmented = np.concatenate([time_channel, path_values], axis=-1)  # (P, T+1, 1+D)
    elif aug_type == "fourier":
        sin_col = np.sin(time_col)[:, :, None]
        cos_col = (np.cos(time_col) - 1.0)[:, :, None]
        augmented = np.concatenate([time_channel, sin_col, cos_col, path_values], axis=-1)  # (P, T+1, 3+D)
    elif aug_type == "TLL":
        augmented = lead_lag_transform(path_values, t=time_grid, include_time=True)
    elif aug_type == "TLL3":
        lead_lag = lead_lag_transform(path_values, t=time_grid, include_time=True)
        augmented = np.concatenate(
            [
                lead_lag[:, :, :1],
                lead_lag[:, :, 1 : 1 + D],
                lead_lag[:, :, 2 + D :],
            ],
            axis=-1,
        )  # drop t^lag
    else:
        raise ValueError(f"Not implemented augmentation: {aug_type}")
        
    return augmented


def compute_logsignatures(
    augmented: np.ndarray, depth: int
) -> np.ndarray:
    """Compute truncated log-signatures of augmented paths.

    Parameters
    ----------
    augmented : (P, T+1, 2) time-augmented paths
    depth     : truncation depth N

    Returns
    -------
    logsigs : (P, D) where D = iisignature.logsigdim(2, depth)
    """
    channels = augmented.shape[2]
    s = iisignature.prepare(channels, depth)
    logsigs = iisignature.logsig(augmented, s)
    return logsigs

def compute_logsignatures_dim(channels: int, depth: int) -> np.ndarray:
    """Return cumulative log-signature lengths for depths 1..N.

    Returns
    -------
    dims : (N,) int array where dims[k-1] = logsiglength(channels, k)
    """
    return np.array([iisignature.logsiglength(channels, k) for k in range(1, depth + 1)])

def compute_logsig_depth_norms(
    logsigs: np.ndarray, channels: int, depth: int
) -> np.ndarray:
    """Compute the L2 norm of log-signature components at each depth level.

    Parameters
    ----------
    logsigs  : (P, D) or (D,) log-signature array (Lyndon basis, ordered by depth)
    channels : path dimension d
    depth    : truncation depth N

    Returns
    -------
    norms : (P, N) array where norms[p, k-1] is the L2 norm of the depth-k
            components of logsigs[p]. Returns (N,) if input was 1-D.
    """
    cum_dims = compute_logsignatures_dim(channels, depth)  # (N,)
    boundaries = np.concatenate([[0], cum_dims])           # (N+1,)

    single = logsigs.ndim == 1
    if single:
        logsigs = logsigs[None, :]

    norms = np.zeros((logsigs.shape[0], depth))
    for k in range(depth):
        norms[:, k] = np.linalg.norm(logsigs[:, boundaries[k] : boundaries[k + 1]], axis=1)

    return norms[0] if single else norms


def compute_signatures(
    augmented: np.ndarray, depth: int
) -> np.ndarray:
    """Compute truncated signatures of augmented paths.

    Parameters
    ----------
    augmented : (P, T+1, 2) time-augmented paths
    depth     : truncation depth N

    Returns
    -------
    sigs : (P, D) where D = iisignature.siglength(2, depth)
    """
    sigs = iisignature.sig(augmented, depth)
    return sigs

def compute_and_save_sigs(
    npz_path: str,
    out_path: str,
    time_grid: np.ndarray,
    sig_depth: int,
    aug_type: str = "TA",
) -> np.ndarray:
    """Compute log-signatures from a paths npz and save to out_path.

    Parameters
    ----------
    npz_path  : .npz file with a "paths" key, shape (N, T), (N, T, D), (N, M, T), or (N, M, T, D)
    out_path  : destination .npz path
    time_grid : (T,) normalized time grid
    sig_depth : truncation depth
    aug_type  : augmentation type used for signature computation

    Returns
    -------
    sigs : (N, S) or (N, M, S) log-signatures
    """
    paths = np.load(npz_path)["paths"]

    if paths.ndim == 4:
        N, M, T, D = paths.shape
        flat = paths.reshape(N * M, T, D)
        aug = augment_path(flat, time_grid, aug_type=aug_type)
        sigs = compute_logsignatures(aug, sig_depth).reshape(N, M, -1)
    elif paths.ndim == 3:
        if time_grid.shape[0] == paths.shape[1]:
            aug = augment_path(paths, time_grid, aug_type=aug_type)
            sigs = compute_logsignatures(aug, sig_depth)
        else:
            N, M, T = paths.shape
            flat = paths.reshape(N * M, T)
            aug = augment_path(flat, time_grid, aug_type=aug_type)
            sigs = compute_logsignatures(aug, sig_depth).reshape(N, M, -1)
    elif paths.ndim == 2:
        aug  = augment_path(paths, time_grid, aug_type=aug_type)
        sigs = compute_logsignatures(aug, sig_depth)
    else:
        raise ValueError(
            f"paths must have shape (N,T), (N,T,D), (N,M,T), or (N,M,T,D), got {paths.shape}"
        )

    channels = aug.shape[-1]

    np.savez_compressed(
        out_path,
        sigs=sigs.astype(np.float32),
        sig_depth=np.int32(sig_depth),
        channels=np.int32(channels),
    )
    return sigs


# ---------------------------------------------------------------------------
# Lead-Lag transform
# ---------------------------------------------------------------------------

def lead_lag_transform(
    X: np.ndarray,
    t: np.ndarray = None,
    include_time: bool = True,
) -> np.ndarray:
    """Lead-lag transform (vectorized, no Python loop over batch).

    Parameters
    ----------
    X            : (B, N, d) array
    t            : (N,) time grid; broadcast over the batch when include_time=True
    include_time : if True, prepend t as a channel before the transform

    Returns
    -------
    out : (B, 2N-1, 2d) array (or (B, 2N-1, 2*(d+1)) when include_time=True)
    """
    X = np.asarray(X, dtype=float)
    B, N, d = X.shape

    if include_time:
        if t is None:
            t = np.linspace(0.0, 1.0, N)
        t_col = np.ascontiguousarray(
            np.broadcast_to(
                np.asarray(t, dtype=float)[None, :, None], (B, N, 1)
            )
        )
        X = np.concatenate([t_col, X], axis=2)  # (B, N, d+1)
        d = d + 1

    lead = np.concatenate(
        [X[:, :1, :], np.repeat(X[:, 1:, :], 2, axis=1)], axis=1
    )  # (B, 2N-1, d)
    lag  = np.concatenate(
        [np.repeat(X[:, :-1, :], 2, axis=1), X[:, -1:, :]], axis=1
    )  # (B, 2N-1, d)
    out  = np.concatenate([lead, lag], axis=2)  # (B, 2N-1, 2d)

    assert out.shape == (B, 2 * N - 1, 2 * d), (
        f"Expected ({B}, {2*N-1}, {2*d}), got {out.shape}"
    )
    return out


def compute_signatures_dim(channels: int, depth: int) -> np.ndarray:
    """Return cumulative signature lengths for depths 1..N.

    Returns
    -------
    dims : (N,) int array where dims[k-1] = siglength(channels, k)
    """
    return np.array([iisignature.siglength(channels, k) for k in range(1, depth + 1)])

# ---------------------------------------------------------------------------
# Conversion between signatures and log-signatures
# ---------------------------------------------------------------------------

# Cache for the Lyndon-to-tensor expansion matrices
_lyndon_tensor_cache: dict[tuple[int, int], np.ndarray] = {}


def _get_lyndon_to_tensor_matrix(channels: int, depth: int) -> np.ndarray:
    """Return the (D_sig, D_logsig) matrix mapping Lyndon basis → tensor algebra.

    Computed once per (channels, depth) pair by solving a linear system
    from paired logsig / expanded-logsig evaluations on random paths.
    """
    key = (channels, depth)
    if key in _lyndon_tensor_cache:
        return _lyndon_tensor_cache[key]

    s = iisignature.prepare(channels, depth)
    D_logsig = iisignature.logsiglength(channels, depth)

    rng = np.random.default_rng(12345)
    n_samples = D_logsig + 10

    L_list, X_list = [], []
    for _ in range(n_samples):
        path = rng.standard_normal((30, channels))
        L_list.append(iisignature.logsig(path, s))        # Lyndon basis
        X_list.append(iisignature.logsig(path, s, "X"))   # tensor algebra

    L = np.array(L_list)   # (n_samples, D_logsig)
    X = np.array(X_list)   # (n_samples, D_sig)

    # X = L @ M^T  →  solve for M^T
    M_T, _, _, _ = np.linalg.lstsq(L, X, rcond=None)
    M = M_T.T  # (D_sig, D_logsig)

    _lyndon_tensor_cache[key] = M
    return M


def _split_sig_levels(x: np.ndarray, d: int, N: int) -> list[np.ndarray]:
    """Split a flat signature vector into per-level arrays."""
    levels, offset = [], 0
    for k in range(1, N + 1):
        size = d**k
        levels.append(x[offset : offset + size])
        offset += size
    return levels


def _tensor_multiply(a: np.ndarray, b: np.ndarray, d: int, N: int) -> np.ndarray:
    """Multiply two elements of the truncated tensor algebra T^N(R^d).

    Both *a* and *b* represent levels 1..N (level 0 is implicitly 0).
    """
    a_lvl = _split_sig_levels(a, d, N)
    b_lvl = _split_sig_levels(b, d, N)
    c_lvl = [np.zeros(d**k) for k in range(1, N + 1)]

    for i in range(1, N + 1):
        for j in range(1, N - i + 1):
            c_lvl[i + j - 1] += np.kron(a_lvl[i - 1], b_lvl[j - 1])

    return np.concatenate(c_lvl)


def _tensor_log(sig: np.ndarray, d: int, N: int) -> np.ndarray:
    """Tensor logarithm: log(1 + x) in T^N(R^d).

    *sig* is a signature vector (levels 1..N; level 0 implicitly 1).
    Returns the expanded log-signature (tensor-algebra form).
    """
    x = sig
    result = x.copy()
    power = x.copy()
    for n in range(2, N + 1):
        power = _tensor_multiply(power, x, d, N)
        result += (-1) ** (n + 1) / n * power
    return result


def _tensor_exp(logsig_expanded: np.ndarray, d: int, N: int) -> np.ndarray:
    """Tensor exponential: exp(L) − 1 in T^N(R^d).

    *logsig_expanded* is a log-signature in tensor-algebra form (levels 1..N).
    Returns the signature vector (levels 1..N).
    """
    result = logsig_expanded.copy()
    power = logsig_expanded.copy()
    for n in range(2, N + 1):
        power = _tensor_multiply(power, logsig_expanded, d, N)
        result += power / factorial(n)
    return result


def sig_to_logsig(
    sigs: np.ndarray, channels: int, depth: int
) -> np.ndarray:
    """Convert signature(s) to log-signature(s).

    Parameters
    ----------
    sigs     : (..., D_sig) signature vectors
    channels : path dimension d
    depth    : truncation depth N

    Returns
    -------
    logsigs : (..., D_logsig) log-signature vectors (Lyndon basis)
    """
    M = _get_lyndon_to_tensor_matrix(channels, depth)
    M_pinv = np.linalg.pinv(M)  # (D_logsig, D_sig)

    single = sigs.ndim == 1
    if single:
        sigs = sigs[None, :]

    expanded = np.array(
        [_tensor_log(row, channels, depth) for row in sigs]
    )
    logsigs = expanded @ M_pinv.T

    return logsigs[0] if single else logsigs


def logsig_to_sig(
    logsigs: np.ndarray, channels: int, depth: int
) -> np.ndarray:
    """Convert log-signature(s) to signature(s).

    Parameters
    ----------
    logsigs  : (..., D_logsig) log-signature vectors (Lyndon basis)
    channels : path dimension d
    depth    : truncation depth N

    Returns
    -------
    sigs : (..., D_sig) signature vectors
    """
    M = _get_lyndon_to_tensor_matrix(channels, depth)

    single = logsigs.ndim == 1
    if single:
        logsigs = logsigs[None, :]

    expanded = logsigs @ M.T  # (P, D_sig)
    sigs = np.array(
        [_tensor_exp(row, channels, depth) for row in expanded]
    )

    return sigs[0] if single else sigs
