import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from playground.dataloader import NextTokenPredictionDataset
from playground.trainer import Trainer


@hydra.main(
    config_path="../configs/experiments/pretraining", config_name="pretrain_verdict"
)
def pretrain(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    dataset_config = config.dataset
    tokenizer = instantiate(config.tokenizer)
    train_dataset = NextTokenPredictionDataset(
        path=dataset_config.train.train_file,
        tokenizer=tokenizer,
        max_length=dataset_config.train.max_length,
        stride=dataset_config.train.stride,
    )
    validation_dataset = NextTokenPredictionDataset(
        path=dataset_config.validation.val_file,
        tokenizer=tokenizer,
        max_length=dataset_config.validation.max_length,
        stride=dataset_config.validation.stride,
    )
    trainer = Trainer(
        model=instantiate(config.model),
        tokenizer=tokenizer,
        optimiser_config=config.optimiser,
        trainer_config=config.trainer,
        data_loader_config=config.data_loader,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    pretrain()
