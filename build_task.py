from Config import (DataConfig,
                    ModelConfig,
                    TaskConfig)


def main(task_card: TaskConfig):
    model_card = task_card.model_config
    data_card = task_card.data_config
