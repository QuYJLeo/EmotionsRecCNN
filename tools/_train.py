import os
import json
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import models


def main(config_path):
    """Main function to start training.
    
    Loads configuration, initializes model and dataset, creates trainer, and starts training.

    Args:
        config_path (str): Path to the configuration JSON file
    """
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()
    print(configs)

    model = get_model(configs)
    train_set, val_set = get_dataset(configs)

    from trainers.mydataset_trainer import MyTrainer
    trainer = MyTrainer(model, train_set, val_set, configs)
    trainer.train()


def get_model(configs):
    """Get the model based on configuration.

    Args:
        configs (dict): Configuration dictionary containing architecture name

    Returns:
        torch.nn.Module: Model class specified in configs["arch"]
    """
    return models.__dict__[configs["arch"]]


def get_dataset(configs):
    """Get training and validation datasets.

    Args:
        configs (dict): Configuration dictionary containing data path and settings

    Returns:
        tuple: (train_set, val_set) - Training and validation datasets
    """
    from utils.datasets.mydataset import MyDataset
    train_set = MyDataset("train", configs)
    val_set = MyDataset("val", configs)
    print('train:', len(train_set), 'val:', len(val_set))
    return train_set, val_set


if __name__ == "__main__":
    main("./configs/ck_config.json")
