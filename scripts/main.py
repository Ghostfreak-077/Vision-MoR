import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import VisionTransformer, MoRVisionTransformer
from scripts.training_scripts import train_epoch
from scripts.evaluating_scripts import evaluate
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
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
    
    # Train Standard ViT
    print("\n" + "="*60)
    print("Training Standard Vision Transformer")
    print("="*60)
    
    vit_model = VisionTransformer(**model_config).to(DEVICE)
    vit_optimizer = torch.optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    vit_train_accs = []
    vit_test_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(vit_model, trainloader, vit_optimizer, DEVICE, is_mor=False)
        test_loss, test_acc = evaluate(vit_model, testloader, DEVICE, is_mor=False)
        
        vit_train_accs.append(train_acc)
        vit_test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Train MoR-ViT
    print("\n" + "="*60)
    print("Training MoR Vision Transformer")
    print("="*60)
    
    mor_config = model_config.copy()
    mor_config['num_recursions'] = 3
    mor_config['use_kv_sharing'] = False
    
    mor_model = MoRVisionTransformer(**mor_config).to(DEVICE)
    mor_optimizer = torch.optim.AdamW(mor_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Count parameters
    vit_params = sum(p.numel() for p in vit_model.parameters())
    mor_params = sum(p.numel() for p in mor_model.parameters())
    print(f"\nViT Parameters: {vit_params:,}")
    print(f"MoR-ViT Parameters: {mor_params:,}")
    print(f"Parameter Reduction: {(1 - mor_params/vit_params)*100:.1f}%")
    
    mor_train_accs = []
    mor_test_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(mor_model, trainloader, mor_optimizer, DEVICE, is_mor=True)
        test_loss, test_acc = evaluate(mor_model, testloader, DEVICE, is_mor=True)
        
        mor_train_accs.append(train_acc)
        mor_test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(vit_train_accs, label='ViT Train', marker='o')
    plt.plot(mor_train_accs, label='MoR-ViT Train', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(vit_test_accs, label='ViT Test', marker='o')
    plt.plot(mor_test_accs, label='MoR-ViT Test', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mor_vit_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final ViT Test Accuracy: {vit_test_accs[-1]:.2f}%")
    print(f"Final MoR-ViT Test Accuracy: {mor_test_accs[-1]:.2f}%")
    print("="*60)
    
if __name__ == '__main__':
    main()