import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

from model import AudioResNet, LabelSmoothingCrossEntropy
from data import get_train_test_split
from utils import calculate_accuracy, plot_training_curves

def train_model(num_epochs=30, batch_size=64, learning_rate=0.0005, weight_decay=0.05, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds, test_ds = get_train_test_split()
    train_specs, train_labels = train_ds.get_data()
    test_specs, test_labels = test_ds.get_data()
    
    train_specs = train_specs.to(device)
    train_labels = train_labels.to(device)
    test_specs = test_specs.to(device)
    test_labels = test_labels.to(device)
    
    model = AudioResNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_start in tqdm(range(0, len(train_specs), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_end = min(batch_start + batch_size, len(train_specs))
            batch_specs = train_specs[batch_start:batch_end].unsqueeze(1)
            batch_labels = train_labels[batch_start:batch_end]
            
            if random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                rand_index = torch.randperm(batch_specs.size(0))
                mixed_specs = lam * batch_specs + (1 - lam) * batch_specs[rand_index]
                mixed_labels = lam * torch.nn.functional.one_hot(batch_labels, num_classes=50).float() + \
                              (1 - lam) * torch.nn.functional.one_hot(batch_labels[rand_index], num_classes=50).float()
                batch_specs = mixed_specs
                batch_labels = mixed_labels
            
            optimizer.zero_grad()
            outputs = model(batch_specs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            if batch_labels.dim() > 1:
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(batch_labels, 1)
                correct_predictions += (predicted == true_labels).sum().item()
            else:
                correct_predictions += (torch.max(outputs.data, 1)[1] == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
        
        epoch_loss = running_loss / (len(train_specs) // batch_size)
        epoch_train_acc = 100 * correct_predictions / total_samples
        
        model.eval()
        with torch.no_grad():
            test_correct = 0
            for i in range(0, len(test_specs), batch_size):
                batch_end = min(i + batch_size, len(test_specs))
                batch_specs = test_specs[i:batch_end].unsqueeze(1)
                batch_labels = test_labels[i:batch_end]
                
                outputs = model(batch_specs)
                test_correct += (torch.max(outputs.data, 1)[1] == batch_labels).sum().item()
            
            test_acc = 100 * test_correct / len(test_specs)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        test_accuracies.append(test_acc)
        
        scheduler.step(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={epoch_train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        
        if no_improve_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
    
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    torch.save(model.state_dict(), 'models/audio_resnet.pth')
    
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch
    }
    torch.save(metrics, 'models/training_metrics.pth')
    
    return model

if __name__ == "__main__":
    model = train_model() 