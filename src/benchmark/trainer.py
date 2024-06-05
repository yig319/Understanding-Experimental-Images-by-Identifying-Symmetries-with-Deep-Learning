from tqdm import tqdm
from typing import Callable, List
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Callable],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_every: int,
        model_path: str,
        early_stopping_patience: int = None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_path = model_path
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.epochs_without_improvement = 0

    def train(self, train_dl, epochs, valid_dl=None, cv_dl=None, tracking=False):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.run_epoch(train_dl, mode='train', tracking=tracking)
            if valid_dl:
                early_stop = self.run_epoch(valid_dl, mode='valid', tracking=tracking)
                if cv_dl:
                    _ = self.run_epoch(cv_dl, mode='valid', tracking=tracking)
                if early_stop:
                    print("Early stopping triggered.")
                    break 
                
            if self.save_every and (epoch + 1) % self.save_every == 0:
                self.save_model(epoch + 1)

                
    def validate(self, valid_dl, tracking=False):
        self.run_epoch(valid_dl, mode='valid', tracking=tracking)

    def run_epoch(self, data_loader, mode='train', tracking=False):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_size = len(data_loader.dataset)
        total_loss = 0.0
        metric_values = {metric.__name__: 0.0 for metric in self.metrics}

        for i, batch in enumerate(tqdm(data_loader)):
            loss, batch_metrics = self.run_batch(batch, mode)
            total_loss += loss * batch[0].size(0)
            for metric_name, metric_value in batch_metrics.items():
                metric_values[metric_name] += metric_value * batch[0].size(0)

            if tracking:
                self.log_wandb({f"{mode}_loss": loss, **batch_metrics})

        avg_loss = total_loss / data_size
        avg_metrics = {metric: value / data_size for metric, value in metric_values.items()}
        print(f"{mode.capitalize()}: Loss: {avg_loss:.4f}, " + ", ".join([f"{metric}: {value * 100:.4f}%" for metric, value in avg_metrics.items()]))

        if mode == 'valid' and self.early_stopping_patience is not None:
            primary_metric = list(self.metrics)[0].__name__  # Assuming the first metric is the primary one for early stopping
            if self.best_metric is None or avg_metrics[primary_metric] > self.best_metric:
                self.best_metric = avg_metrics[primary_metric]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    return True  # Indicate early stopping

    def run_batch(self, batch, mode):
        inputs = batch[0].to(self.device).float()
        labels = batch[1].to(self.device).long()

        if mode == 'train':
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'train'):
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            batch_metrics = {metric.__name__: metric(outputs, labels) for metric in self.metrics}

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        return loss.item(), batch_metrics

    
    def run_batch(self, batch, mode):
        inputs = batch[0].to(self.device).float()
        labels = batch[1].to(self.device).long()

        if mode == 'train':
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'train'):
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            batch_metrics = {metric.__name__: metric(outputs, labels) for metric in self.metrics}

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        return loss.item(), batch_metrics


    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.model_path}/model_epoch_{epoch}.pth")
        print(f"Model saved at epoch {epoch}")

    def log_wandb(self, metrics):
        wandb.log(metrics)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Example model, loss function, optimizer, and metrics
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = [accuracy]

    # Create and run the trainer
    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=metrics,
        scheduler=None,
        device=device,
        save_every=5,
        model_path='./'
    )
    trainer.train(train_loader, epochs=10)

    # Clean up the downloaded MNIST dataset and saved model
    shutil.rmtree('./data', ignore_errors=True)
    for epoch in range(1, 11, 5):  # Assuming epochs=10 and save_every=5
        model_file = f'./model_epoch_{epoch}.pth'
        if os.path.exists(model_file):
            os.remove(model_file)
    print("Data and saved models are cleaned up.")

if __name__ == "__main__":
    main()