from importlib import resources

from omegaconf import DictConfig, OmegaConf
from transformers import GPT2LMHeadModel

from playground.checkpoint_utils import load_hf_weights, resolve_config_to_hf_repo_name
from playground.metadata import PACKAGE_NAME


def load_model_config(model_name: str) -> DictConfig:
    # TODO: IF CONFIG WILL BE STORED IN A HIERARCHY ROOTED
    #  AT CONFIG, THEN UPDATE TO RECURSE THROUGH THE HIERARCHY
    config_dir = resources.files(f"{PACKAGE_NAME}.config")
    try:
        cfg = OmegaConf.load(config_dir / "models" / f"{model_name}.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"No config matched {model_name}!")
    return cfg


class TransformerMixin:

    @classmethod
    def from_hf_weights(cls, model_name: str, config: DictConfig | None = None):

        if config is None:
            config = load_model_config(model_name)

        model = cls(config)
        hf_model = GPT2LMHeadModel.from_pretrained(
            resolve_config_to_hf_repo_name(model_name)
        )
        load_hf_weights(model, hf_model.state_dict())
        return model
