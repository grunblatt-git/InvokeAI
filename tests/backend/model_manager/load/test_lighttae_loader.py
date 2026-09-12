from pathlib import Path
from unittest.mock import patch

import torch

from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Anima_Config, VAE_Checkpoint_LightTAE_Config
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader
from invokeai.backend.krea2.lighttae import LightTAE
from invokeai.backend.model_manager.taxonomy import ModelSourceType
from invokeai.backend.krea2.vae_compat import as_qwen_image_vae, patch_qwen_image_vae_tiling


def test_lighttae_config_recognizes_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "lighttaew2_1.safetensors"
    checkpoint_path.touch()
    config = VAE_Checkpoint_LightTAE_Config.from_model_on_disk(
        type("Model", (), {"path": checkpoint_path, "load_state_dict": lambda self: {
            "encoder.0.weight": torch.empty(64, 3, 3, 3),
            "decoder.22.weight": torch.empty(3, 64, 3, 3),
            "decoder.22.bias": torch.empty(3),
        }})(),
        {
            "hash": "hash",
            "path": str(checkpoint_path),
            "file_size": 1,
            "name": "lighttaew2_1",
            "source": str(checkpoint_path),
            "source_type": ModelSourceType.Path,
        },
    )
    assert config.base.value == "qwen-image"


def test_lighttae_matches_checkpoint_layout() -> None:
    model = LightTAE()
    state_dict = model.state_dict()
    assert state_dict["encoder.0.weight"].shape == (64, 3, 3, 3)
    assert state_dict["decoder.22.bias"].shape == (3,)
    assert model.config.z_dim == 16


def test_lighttae_loader_uses_strict_state_dict(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "lightaew2_1.safetensors"
    checkpoint.touch()
    state_dict = LightTAE().state_dict()
    loader = VAELoader.__new__(VAELoader)
    loader._ram_cache = type("Cache", (), {"make_room": lambda self, _: None})()
    loader._torch_dtype = torch.float32
    config = VAE_Checkpoint_LightTAE_Config.model_construct(path=str(checkpoint))

    with patch("safetensors.torch.load_file", return_value=state_dict):
        model = loader._load_lighttae(config)

    assert isinstance(model, LightTAE)
    assert model.encoder[0].weight.device.type == "cpu"
    assert model.config.latents_mean[0] == -0.7571
    assert model.config.latents_std[0] == 2.8184


def test_lighttae_uses_qwen_compatibility_interface() -> None:
    model = LightTAE()
    assert as_qwen_image_vae(model) is model
    with patch_qwen_image_vae_tiling(model, tile_size=None):
        pass


def test_stale_anima_config_is_redirected_to_lighttae_loader(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lighttaew2_1.safetensors"
    checkpoint.touch()
    loader = VAELoader.__new__(VAELoader)
    loader._ram_cache = type("Cache", (), {"make_room": lambda self, _: None})()
    loader._torch_dtype = torch.float32
    config = VAE_Checkpoint_Anima_Config.model_construct(path=str(checkpoint))

    with patch("safetensors.torch.load_file", return_value=LightTAE().state_dict()):
        model = loader._load_model(config)

    assert isinstance(model, LightTAE)