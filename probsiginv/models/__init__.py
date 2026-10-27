from .unet_1d import UNet1D
from .transformer_1d import DiT1D


def _normalize_arch(arch: str) -> str:
    token = str(arch).strip().lower()
    if token == "transformer":
        return "dit"
    return token


def build_model(arch: str, seq_len: int, cond_dim: int, **kwargs):
    """Instantiate a noise-prediction model by architecture name.

    Parameters
    ----------
    arch     : "unet" | "dit" (legacy alias: "transformer")
    seq_len  : input sequence length
    cond_dim : conditioning vector dimension
    **kwargs : architecture-specific hyperparameters
    """
    arch = _normalize_arch(arch)
    path_dim = int(kwargs.get("path_dim", 1))
    if path_dim < 1:
        raise ValueError(f"path_dim must be positive, got {path_dim}")

    if arch == "unet":
        return UNet1D(
            seq_len=seq_len,
            cond_dim=cond_dim,
            path_dim=path_dim,
            channels=kwargs.get("channels", [32, 64, 128, 128]),
            num_res_blocks=kwargs.get("num_res_blocks", 2),
            emb_dim=kwargs.get("emb_dim", 128),
        )
    elif arch == "dit":
        return DiT1D(
            seq_len=seq_len,
            cond_dim=cond_dim,
            path_dim=path_dim,
            patch_size=kwargs.get("patch_size", 8),
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 8),
            n_layers=kwargs.get("n_layers", 7),
            emb_dim=kwargs.get("emb_dim", 128),
            final_deblock=kwargs.get("final_deblock", False),
        )
    else:
        raise ValueError(f"Unknown arch: '{arch}'. Choose 'unet' or 'dit'.")


__all__ = ["UNet1D", "DiT1D", "build_model"]
