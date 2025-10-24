from collections import defaultdict
import time
import torch
from torch import nn
from models import MoRVisionTransformer
from models import VisionTransformer
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MetricsTracker:
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        
    def get_model_params(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'params_mb': total_params * 4 / (1024**2)  # Assuming fp32
        }
    
    def get_memory_stats(self):
        """Get GPU memory statistics"""
        if self.device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.memory_allocated(self.device) / (1024**2),
                'reserved_mb': torch.cuda.memory_reserved(self.device) / (1024**2),
                'max_allocated_mb': torch.cuda.max_memory_allocated(self.device) / (1024**2),
                'max_reserved_mb': torch.cuda.max_memory_reserved(self.device) / (1024**2)
            }
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0, 'max_reserved_mb': 0}
    
    def measure_inference_time(self, model, input_size, batch_size=32, num_runs=100):
        """Measure average inference latency"""
        model.eval()
        dummy_input = torch.randn(batch_size, *input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if isinstance(model, MoRVisionTransformer):
                    _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
        
        # Measure
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                
                if isinstance(model, MoRVisionTransformer):
                    _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append(time.perf_counter() - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(times)
        }
    
    def measure_flops(self, model, input_size):
        """Estimate FLOPs (simplified)"""
        # This is a rough estimation
        dummy_input = torch.randn(1, *input_size).to(self.device)
        total_ops = 0
        
        def count_ops_hook(module, input, output):
            nonlocal total_ops
            if isinstance(module, nn.Linear):
                total_ops += input[0].numel() * module.weight.size(0)
            elif isinstance(module, nn.Conv2d):
                batch_size, in_c, in_h, in_w = input[0].size()
                out_c, _, k_h, k_w = module.weight.size()
                out_h = (in_h + 2 * module.padding[0] - k_h) // module.stride[0] + 1
                out_w = (in_w + 2 * module.padding[1] - k_w) // module.stride[1] + 1
                total_ops += batch_size * out_c * out_h * out_w * in_c * k_h * k_w
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(count_ops_hook))
        
        with torch.no_grad():
            if isinstance(model, MoRVisionTransformer):
                _ = model(dummy_input)
            else:
                _ = model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return {
            'flops': total_ops,
            'gflops': total_ops / 1e9
        }
        
def compare():
    # Hyperparameters
    BATCH_SIZE = 512
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print(f"{'MIXTURE-OF-RECURSIONS VISION TRANSFORMER TRAINING':^80}")
    print("="*80)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("="*80)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(DEVICE)
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model configurations
    model_config = {
        'img_size': 32,
        'patch_size': 4,
        'in_channels': 3,
        'num_classes': 10,
        'embed_dim': 256,
        'depth': 6,
        'num_heads': 8,
        'mlp_ratio': 4.0
    }
    
    # ========================================================================
    # Train Standard ViT
    # ========================================================================
    print("\n" + "="*80)
    print(f"{'STANDARD VISION TRANSFORMER':^80}")
    print("="*80)
    
    vit_model = VisionTransformer(**model_config).to(DEVICE)
    vit_optimizer = torch.optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Get model statistics
    vit_params = tracker.get_model_params(vit_model)
    print(f"\n{'Model Statistics':^80}")
    print("-"*80)
    print(f"Total Parameters: {vit_params['total_params']:,}")
    print(f"Trainable Parameters: {vit_params['trainable_params']:,}")
    print(f"Model Size: {vit_params['params_mb']:.2f} MB")
    
    # Measure inference latency
    print(f"\n{'Inference Metrics':^80}")
    print("-"*80)
    vit_inference = tracker.measure_inference_time(vit_model, (3, 32, 32), BATCH_SIZE)
    print(f"Mean Latency: {vit_inference['mean_latency_ms']:.2f} ± {vit_inference['std_latency_ms']:.2f} ms")
    print(f"Throughput: {vit_inference['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Measure FLOPs
    vit_flops = tracker.measure_flops(vit_model, (3, 32, 32))
    print(f"GFLOPs per sample: {vit_flops['gflops']:.2f}")
    
    vit_train_history = []
    vit_test_history = []
    
    print(f"\n{'Training Progress':^80}")
    print("-"*80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*40)
        
        train_metrics = train_epoch(vit_model, trainloader, vit_optimizer, DEVICE, 
                                    is_mor=False, tracker=tracker)
        test_metrics = evaluate(vit_model, testloader, DEVICE, 
                               is_mor=False, tracker=tracker)
        
        vit_train_history.append(train_metrics)
        vit_test_history.append(test_metrics)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
              f"Time: {train_metrics['mean_batch_time_ms']:.2f}ms/batch")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, "
              f"Time: {test_metrics['mean_batch_time_ms']:.2f}ms/batch")
        
        if DEVICE.type == 'cuda':
            print(f"Peak Memory: {train_metrics['max_allocated_mb']:.2f} MB")
    
    # ========================================================================
    # Train MoR-ViT
    # ========================================================================
    print("\n" + "="*80)
    print(f"{'MIXTURE-OF-RECURSIONS VISION TRANSFORMER':^80}")
    print("="*80)
    
    mor_config = model_config.copy()
    mor_config['num_recursions'] = 3
    mor_config['use_kv_sharing'] = False
    
    mor_model = MoRVisionTransformer(**mor_config).to(DEVICE)
    mor_optimizer = torch.optim.AdamW(mor_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Get model statistics
    mor_params = tracker.get_model_params(mor_model)
    print(f"\n{'Model Statistics':^80}")
    print("-"*80)
    print(f"Total Parameters: {mor_params['total_params']:,}")
    print(f"Trainable Parameters: {mor_params['trainable_params']:,}")
    print(f"Model Size: {mor_params['params_mb']:.2f} MB")
    print(f"Parameter Reduction: {(1 - mor_params['total_params']/vit_params['total_params'])*100:.2f}%")
    print(f"Size Reduction: {(1 - mor_params['params_mb']/vit_params['params_mb'])*100:.2f}%")
    
    # Measure inference latency
    print(f"\n{'Inference Metrics':^80}")
    print("-"*80)
    mor_inference = tracker.measure_inference_time(mor_model, (3, 32, 32), BATCH_SIZE)
    print(f"Mean Latency: {mor_inference['mean_latency_ms']:.2f} ± {mor_inference['std_latency_ms']:.2f} ms")
    print(f"Throughput: {mor_inference['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Speedup: {mor_inference['throughput_samples_per_sec']/vit_inference['throughput_samples_per_sec']:.2f}x")
    
    # Measure FLOPs
    mor_flops = tracker.measure_flops(mor_model, (3, 32, 32))
    print(f"GFLOPs per sample: {mor_flops['gflops']:.2f}")
    print(f"FLOPs Reduction: {(1 - mor_flops['gflops']/vit_flops['gflops'])*100:.2f}%")
    
    mor_train_history = []
    mor_test_history = []
    
    print(f"\n{'Training Progress':^80}")
    print("-"*80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*40)
        
        train_metrics = train_epoch(mor_model, trainloader, mor_optimizer, DEVICE, 
                                    is_mor=True, tracker=tracker)
        test_metrics = evaluate(mor_model, testloader, DEVICE, 
                               is_mor=True, tracker=tracker)
        
        mor_train_history.append(train_metrics)
        mor_test_history.append(test_metrics)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
              f"Aux: {train_metrics['aux_loss']:.4f}, Time: {train_metrics['mean_batch_time_ms']:.2f}ms/batch")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, "
              f"Time: {test_metrics['mean_batch_time_ms']:.2f}ms/batch")
        
        if DEVICE.type == 'cuda':
            print(f"Peak Memory: {train_metrics['max_allocated_mb']:.2f} MB")
    
    # ========================================================================
    # Final Comparison
    # ========================================================================
    print("\n" + "="*80)
    print(f"{'FINAL COMPARISON':^80}")
    print("="*80)
    
    print(f"\n{'Model Efficiency Metrics':^80}")
    print("-"*80)
    print(f"{'Metric':<40} {'ViT':>15} {'MoR-ViT':>15} {'Improvement':>10}")
    print("-"*80)
    
    print(f"{'Parameters (M)':<40} {vit_params['total_params']/1e6:>15.2f} "
          f"{mor_params['total_params']/1e6:>15.2f} "
          f"{(1-mor_params['total_params']/vit_params['total_params'])*100:>9.1f}%")
    
    print(f"{'Model Size (MB)':<40} {vit_params['params_mb']:>15.2f} "
          f"{mor_params['params_mb']:>15.2f} "
          f"{(1-mor_params['params_mb']/vit_params['params_mb'])*100:>9.1f}%")
    
    print(f"{'Inference Latency (ms)':<40} {vit_inference['mean_latency_ms']:>15.2f} "
          f"{mor_inference['mean_latency_ms']:>15.2f} "
          f"{(1-mor_inference['mean_latency_ms']/vit_inference['mean_latency_ms'])*100:>9.1f}%")
    
    print(f"{'Throughput (samples/s)':<40} {vit_inference['throughput_samples_per_sec']:>15.2f} "
          f"{mor_inference['throughput_samples_per_sec']:>15.2f} "
          f"{(mor_inference['throughput_samples_per_sec']/vit_inference['throughput_samples_per_sec']-1)*100:>9.1f}%")
    
    print(f"{'GFLOPs':<40} {vit_flops['gflops']:>15.2f} "
          f"{mor_flops['gflops']:>15.2f} "
          f"{(1-mor_flops['gflops']/vit_flops['gflops'])*100:>9.1f}%")
    
    vit_final_acc = vit_test_history[-1]['accuracy']
    mor_final_acc = mor_test_history[-1]['accuracy']
    
    print(f"{'Final Test Accuracy (%)':<40} {vit_final_acc:>15.2f} "
          f"{mor_final_acc:>15.2f} "
          f"{(mor_final_acc-vit_final_acc):>9.2f}%")
    
    if DEVICE.type == 'cuda':
        vit_peak_mem = max(m['max_allocated_mb'] for m in vit_train_history)
        mor_peak_mem = max(m['max_allocated_mb'] for m in mor_train_history)
        print(f"{'Peak Training Memory (MB)':<40} {vit_peak_mem:>15.2f} "
              f"{mor_peak_mem:>15.2f} "
              f"{(1-mor_peak_mem/vit_peak_mem)*100:>9.1f}%")
    
    print("-"*80)
    
    # ========================================================================
    # Plot Results
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, EPOCHS + 1)
    
    # Training Accuracy
    axes[0, 0].plot(epochs, [m['accuracy'] for m in vit_train_history], 
                    label='ViT', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, [m['accuracy'] for m in mor_train_history], 
                    label='MoR-ViT', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[0, 1].plot(epochs, [m['accuracy'] for m in vit_test_history], 
                    label='ViT', marker='o', linewidth=2)
    axes[0, 1].plot(epochs, [m['accuracy'] for m in mor_test_history], 
                    label='MoR-ViT', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Loss
    axes[0, 2].plot(epochs, [m['loss'] for m in vit_train_history], 
                    label='ViT', marker='o', linewidth=2)
    axes[0, 2].plot(epochs, [m['loss'] for m in mor_train_history], 
                    label='MoR-ViT', marker='s', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Loss', fontsize=12)
    axes[0, 2].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Batch Time
    axes[1, 0].plot(epochs, [m['mean_batch_time_ms'] for m in vit_train_history], 
                    label='ViT', marker='o', linewidth=2)
    axes[1, 0].plot(epochs, [m['mean_batch_time_ms'] for m in mor_train_history], 
                    label='MoR-ViT', marker='s', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Time (ms)', fontsize=12)
    axes[1, 0].set_title('Mean Batch Training Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Throughput
    axes[1, 1].plot(epochs, [m['samples_per_sec'] for m in vit_train_history], 
                    label='ViT', marker='o', linewidth=2)
    axes[1, 1].plot(epochs, [m['samples_per_sec'] for m in mor_train_history], 
                    label='MoR-ViT', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Samples/sec', fontsize=12)
    axes[1, 1].set_title('Training Throughput', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Model Comparison Bar Chart
    metrics_names = ['Params\n(M)', 'Size\n(MB)', 'Latency\n(ms)', 'GFLOPs']
    vit_values = [
        vit_params['total_params']/1e6,
        vit_params['params_mb'],
        vit_inference['mean_latency_ms'],
        vit_flops['gflops']
    ]
    mor_values = [
        mor_params['total_params']/1e6,
        mor_params['params_mb'],
        mor_inference['mean_latency_ms'],
        mor_flops['gflops']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize for better visualization
    vit_norm = [v/m for v, m in zip(vit_values, vit_values)]
    mor_norm = [v/m for v, m in zip(mor_values, vit_values)]
    
    axes[1, 2].bar(x - width/2, vit_norm, width, label='ViT', alpha=0.8)
    axes[1, 2].bar(x + width/2, mor_norm, width, label='MoR-ViT', alpha=0.8)
    axes[1, 2].set_ylabel('Normalized Value', fontsize=12)
    axes[1, 2].set_title('Model Efficiency Comparison\n(Normalized to ViT)', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics_names)
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('mor_vit_comprehensive_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to 'mor_vit_comprehensive_results.png'")
    plt.show()
    
if __name__ == "__main__":
    compare()