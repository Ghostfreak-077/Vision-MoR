import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.mor_model import MoRViTModel
from scripts.training_scripts import train_epoch
from scripts.evaluating_scripts import evaluate
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTConfig

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
    config = ViTConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=256 * 4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=32,
        patch_size=4,
        num_channels=3,
        num_labels=10,
        # MoR specific config
        num_recursions=3,
    )

    vit_model = ViTModel(config).to(DEVICE)
    vit_optimizer = torch.optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    classifier = nn.Linear(config.hidden_size, config.num_labels).to(DEVICE)
    
    vit_train_accs = []
    vit_test_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(vit_model, trainloader, vit_optimizer, classifier, DEVICE, is_mor=False)
        test_loss, test_acc = evaluate(vit_model, testloader, DEVICE, is_mor=False)
        
        vit_train_accs.append(train_acc)
        vit_test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Train MoR-ViT
    print("\n" + "="*60)
    print("Training MoR Vision Transformer")
    print("="*60)

    mor_model = MoRViTModel(config).to(DEVICE)
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
        train_loss, train_acc = train_epoch(mor_model, trainloader, mor_optimizer, classifier, DEVICE, is_mor=True)
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