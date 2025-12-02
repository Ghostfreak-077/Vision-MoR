import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train_epoch(model, loader, optimizer, classifier, device, is_pretrained=False, tracker=None):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    correct = 0
    total = 0
    batch_times = []
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        batch_start = time.perf_counter()
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        if is_pretrained:
            logits = model(images).logits
        else:
            last_hidden_state = model(images).last_hidden_state
            cls = last_hidden_state[:, 0]  # (B, hidden_size)
            logits = classifier(cls)

        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'batch_time': f'{batch_time*1000:.1f}ms'
            })
    
    # Get memory stats
    mem_stats = tracker.get_memory_stats() if tracker else {}
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100. * correct / total,
        'aux_loss': total_aux_loss / len(loader),
        'mean_batch_time_ms': np.mean(batch_times) * 1000,
        'samples_per_sec': len(loader.dataset) / sum(batch_times),
        **mem_stats
    }
