import torch
from tqdm import tqdm
import wandb
from typing import Callable, List

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

    def train(self, train_loader, epochs, valid_loader=None, cv_loader=None, tracking=False):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.run_epoch(train_loader, mode='train', tracking=tracking)
            if valid_loader:
                early_stop = self.run_epoch(valid_loader, mode='valid', tracking=tracking)
                if cv_loader:
                    _ = self.run_epoch(cv_loader, mode='valid', tracking=tracking)
                if early_stop:
                    print("Early stopping triggered.")
                    break 
                
            if self.save_every and (epoch + 1) % self.save_every == 0:
                self.save_model(epoch + 1)

                
    def validate(self, valid_loader, tracking=False):
        self.run_epoch(valid_loader, mode='valid', tracking=tracking)

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



# class Trainer:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         loss_func: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         metrics: List[Callable],
#         scheduler: torch.optim.lr_scheduler._LRScheduler,
#         device: torch.device,
#         save_every: int,
#         model_path: str,
#         early_stopping_patience: int = None,
#     ) -> None:
#         self.device = device
#         self.model = model.to(device)
#         self.loss_func = loss_func
#         self.optimizer = optimizer
#         self.metrics = metrics
#         self.scheduler = scheduler
#         self.save_every = save_every
#         self.model_path = model_path
#         self.early_stopping_patience = early_stopping_patience
#         self.best_metric = None
#         self.epochs_without_improvement = 0


    
#     def run_epoch(self, data_loader, mode='train', tracking=False):
#         # Update to handle multiple metrics
#         metric_values = {metric.__name__: 0.0 for metric in self.metrics}

#         # Rest of the method remains the same...

#         # Update metric_values based on batch results

#         # Check for early stopping
#         if mode == 'val' and self.early_stopping_patience is not None:
#             current_metric = metric_values.get('accuracy', 0.0)  # Example: using accuracy as the metric
#             if self.best_metric is None or current_metric > self.best_metric:
#                 self.best_metric = current_metric
#                 self.epochs_without_improvement = 0
#             else:
#                 self.epochs_without_improvement += 1
#                 if self.epochs_without_improvement >= self.early_stopping_patience:
#                     print("Early stopping triggered")
#                     return True  # Indicate that early stopping was triggered

#         return False  # Indicate that training should continue

#     def run_batch(self, batch, mode):
#         inputs = batch[0].to(self.device).float()
#         labels = batch[1].to(self.device).long()

#         if mode == 'train':
#             self.optimizer.zero_grad()

#         with torch.set_grad_enabled(mode == 'train'):
#             outputs = self.model(inputs)
#             loss = self.loss_func(outputs, labels)
#             _, predictions = torch.max(outputs.data, 1)
#             correct_counts = predictions.eq(labels.data.view_as(predictions))
#             acc = torch.mean(correct_counts.type(torch.FloatTensor))

#             if mode == 'train':
#                 loss.backward()
#                 self.optimizer.step()
#                 if self.scheduler:
#                     self.scheduler.step()

#         return loss.item(), acc.item()

#     def save_model(self, epoch):
#         if epoch % self.save_every == 0:
#             torch.save(self.model.state_dict(), f"{self.model_path}/model_epoch_{epoch}.pth")
#             print(f"Model saved at epoch {epoch}")

#     def log_wandb(self, metrics):
#         wandb.log(metrics)
