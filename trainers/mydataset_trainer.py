import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.radam import RAdam
from utils.metrics.metrics import accuracy

class Trainer(object):
    """Base class for all trainers.
    
    Provides common functionality and interface for training loops.
    """

    def __init__(self):
        pass


class MyTrainer(Trainer):
    """Trainer class for emotion recognition model training.
    
    Handles training loop, validation, logging, and checkpoint saving.

    Args:
        model (torch.nn.Module): Model class to instantiate and train
        train_set (torch.utils.data.Dataset): Training dataset
        val_set (torch.utils.data.Dataset): Validation dataset
        configs (dict): Configuration dictionary with training parameters
    """
    def __init__(self, model, train_set, val_set, configs):
        super().__init__()
        print("Start trainer..")

        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"]
        self._weight_decay = self._configs["weight_decay"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]

        self._train_set = train_set
        self._val_set = val_set
        self._model = model(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
            weight_path=configs["weight_path"],
        )

        self._model = self._model.to(self._device)

        self._train_loader = DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self._val_loader = DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
        )

        self._criterion = nn.CrossEntropyLoss().to(self._device)

        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            patience=self._configs["plateau_patience"],
            min_lr=1e-6,
            verbose=True,
        )

        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        self._train_loss_list = []
        self._train_acc_list = []
        self._val_loss_list = []
        self._val_acc_list = []
        self._best_val_loss = 1e9
        self._best_val_acc = 0
        self._best_train_loss = 1e9
        self._best_train_acc = 0
        self._plateau_count = 0
        self._current_epoch_num = 0

        self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}".format(
                self._configs["arch"], self._configs["model_name"]
            ),
        )

    def _train(self):
        """Perform one epoch of training.
        
        Iterates over the training dataset, computes loss, and updates model weights.
        """
        self._model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, (images, targets) in tqdm(
            enumerate(self._train_loader), total=len(self._train_loader), leave=False
        ):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            outputs = self._model(images)
            loss = self._criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]

            train_loss += loss.item()
            train_acc += acc.item()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        i += 1
        self._train_loss_list.append(train_loss / i)
        self._train_acc_list.append(train_acc / i)

    def _val(self):
        """Perform one epoch of validation.
        
        Iterates over the validation dataset and computes loss and accuracy without updating weights.
        """
        self._model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._val_loader), total=len(self._val_loader), leave=False
            ):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                outputs = self._model(images)
                loss = self._criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]

                val_loss += loss.item()
                val_acc += acc.item()

            i += 1
            self._val_loss_list.append(val_loss / i)
            self._val_acc_list.append(val_acc / i)

    def train(self):
        """Main training loop.
        
        Runs training and validation epochs until stopping criteria are met.
        """
        while not self._is_stop():
            self._increase_epoch_num()
            self._train()
            self._val()

            self._update_training_state()
            self._logging()

            consume_time = str(datetime.datetime.now() - self._start_time)
            print("Summary", "Converged after {} epochs, consume {}".format(self._current_epoch_num, consume_time[:-7]))
            print("Results", "Best validation accuracy: {:.3f}".format(self._best_val_acc))
            print("Results", "Best training accuracy: {:.3f}".format(self._best_train_acc))

    def _update_training_state(self):
        """Update training state including best metrics and plateau counter."""
        if self._val_acc_list[-1] > self._best_val_acc:
            self._save_weights()
            self._best_val_acc = self._val_acc_list[-1]
            self._best_val_loss = self._val_loss_list[-1]
            self._best_train_acc = self._train_acc_list[-1]
            self._best_train_loss = self._train_loss_list[-1]
            self._plateau_count = 0
        else:
            self._plateau_count += 1

        self._scheduler.step(100 - self._val_acc_list[-1])

    def _logging(self):
        """Log training progress including loss and accuracy metrics."""
        consume_time = str(datetime.datetime.now() - self._start_time)
        message = "\nEpoch{:03d}  loss: train{:.3f}/val{:.3f}/best{:.3f}  acc:train{:.3f}/val{:.3f}/best{:.3f} | p{:02d}  Time {}\n".format(
            self._current_epoch_num,
            self._train_loss_list[-1],
            self._val_loss_list[-1],
            self._best_val_loss,
            self._train_acc_list[-1],
            self._val_acc_list[-1],
            self._best_val_acc,
            self._plateau_count,
            consume_time[:-7],
        )
        print(message)

    def _is_stop(self):
        """Check if training should stop.
        
        Returns:
            bool: True if training should stop (plateau count exceeded or max epochs reached)
        """
        return (
            self._plateau_count > self._max_plateau_count
            or self._current_epoch_num > self._max_epoch_num
        )

    def _increase_epoch_num(self):
        """Increment the current epoch number."""
        self._current_epoch_num += 1

    def _save_weights(self):
        """Save model checkpoint to disk."""
        state_dict = self._model.state_dict()
        state = {
            "net": state_dict,
        }
        torch.save(state, self._checkpoint_path + '.pth')
