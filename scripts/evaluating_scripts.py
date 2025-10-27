import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def evaluate(model, loader, device, is_mor=False, tracker=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    batch_times = []
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            batch_start = time.perf_counter()
            
            images, labels = images.to(device), labels.to(device)
            
            if is_mor:
                logits, _ = model(images)
            else:
                logits = model(images)
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            batch_times.append(time.perf_counter() - batch_start)
    
    # Get memory stats
    mem_stats = tracker.get_memory_stats() if tracker else {}
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100. * correct / total,
        'mean_batch_time_ms': np.mean(batch_times) * 1000,
        'samples_per_sec': len(loader.dataset) / sum(batch_times),
        **mem_stats
    }