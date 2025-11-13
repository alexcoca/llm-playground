import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, float], step: int):
        """Log metrics at a given step."""
        pass

    def close(self):
        """Cleanup (e.g., wandb.finish())."""
        pass


class ConsoleLogger(Logger):
    def log(self, metric: dict[str, float], step: int):
        for metric_key, metric_value in metric.items():
            logger.info(f"Step {step}: {metric_key}: {metric_value}")


# class WandbLogger(Logger):
#     def __init__(self, project, name, config):
#         wandb.init(project=project, name=name, config=config)
#
#     def log(self, metrics, step):
#         wandb.log(metrics, step=step)
#
#     def close(self):
#         wandb.finish()
