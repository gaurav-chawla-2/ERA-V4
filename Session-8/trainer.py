"""
Training utilities for ResNet CIFAR-100
Implements training loop, validation, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from collections import defaultdict
from data_loader import mixup_data, mixup_criterion

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Choose scheduler based on config
        if hasattr(config, 'USE_COSINE_ANNEALING') and config.USE_COSINE_ANNEALING:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.COSINE_T_MAX,
                eta_min=config.COSINE_ETA_MIN
            )
        else:
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=config.LR_SCHEDULE,
                gamma=config.LR_GAMMA
            )
        
        # Loss function with label smoothing
        if hasattr(config, 'LABEL_SMOOTHING') and config.LABEL_SMOOTHING > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup if enabled
            if hasattr(self.config, 'MIXUP') and self.config.MIXUP and np.random.rand() < 0.5:
                data, target_a, target_b, lam = mixup_data(data, target, self.config.MIXUP_ALPHA)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                
                # Calculate accuracy (use original target for simplicity)
                pred = output.argmax(dim=1)
                correct = lam * pred.eq(target_a).sum().item() + (1 - lam) * pred.eq(target_b).sum().item()
                accuracy = 100.0 * correct / target.size(0)
            else:
                # Standard training
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct = pred.eq(target).sum().item()
                accuracy = 100.0 * correct / target.size(0)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update meters
            losses.update(loss.item(), data.size(0))
            top1.update(accuracy, data.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg, top1.avg
    
    def validate(self, data_loader, dataset_name="Validation"):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc=f'{dataset_name}'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)  # Use standard CE for validation
                
                # Calculate top-1 accuracy
                pred = output.argmax(dim=1)
                correct = pred.eq(target).sum().item()
                accuracy = 100.0 * correct / target.size(0)
                
                # Calculate top-5 accuracy
                _, pred5 = output.topk(5, 1, True, True)
                pred5 = pred5.t()
                correct5 = pred5.eq(target.view(1, -1).expand_as(pred5)).sum().item()
                accuracy5 = 100.0 * correct5 / target.size(0)
                
                losses.update(loss.item(), data.size(0))
                top1.update(accuracy, data.size(0))
                top5.update(accuracy5, data.size(0))
        
        return losses.avg, top1.avg, top5.avg
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest.pth')
        torch.save(state, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best.pth')
            torch.save(state, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            
            print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, best val acc: {self.best_val_acc:.2f}%)")
            return checkpoint['epoch']
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")
            return 0
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plots
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.axhline(y=self.config.TARGET_ACCURACY, color='g', linestyle='--', 
                   label=f'Target ({self.config.TARGET_ACCURACY}%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.learning_rates, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Best accuracy tracking
        best_accs = [max(self.val_accuracies[:i+1]) for i in range(len(self.val_accuracies))]
        ax4.plot(epochs, best_accs, 'purple', linewidth=2)
        ax4.axhline(y=self.config.TARGET_ACCURACY, color='g', linestyle='--', 
                   label=f'Target ({self.config.TARGET_ACCURACY}%)')
        ax4.set_title('Best Validation Accuracy Progress')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Best Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, start_epoch=0):
        """Main training loop"""
        print(f"Starting training from epoch {start_epoch + 1}")
        print(f"Target accuracy: {self.config.TARGET_ACCURACY}%")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_acc5 = self.validate(self.val_loader, "Validation")
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            if self.config.SAVE_MODEL:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5: {val_acc5:.2f}%")
            print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
            print(f"Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if target reached
            if val_acc >= self.config.TARGET_ACCURACY:
                print(f"\n🎉 Target accuracy of {self.config.TARGET_ACCURACY}% reached!")
                print(f"Achieved {val_acc:.2f}% at epoch {epoch+1}")
        
        # Final test evaluation
        print("\nFinal evaluation on test set:")
        test_loss, test_acc, test_acc5 = self.validate(self.test_loader, "Test")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Top-1 Accuracy: {test_acc:.2f}%")
        print(f"Test Top-5 Accuracy: {test_acc5:.2f}%")
        
        # Plot training history
        self.plot_training_history()
        
        return self.best_val_acc, test_acc