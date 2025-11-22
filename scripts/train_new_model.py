import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.mor_model import MoRViTModel
from scripts.training_scripts import train_epoch
from scripts.evaluating_scripts import evaluate
from transformers import ViTConfig
import matplotlib.pyplot as plt
import torch.nn as nn

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
    
    # Model configuration
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
    
    # Train MoR-ViT
    print("\n" + "="*60)
    print("Training MoR Vision Transformer (new_model.py)")
    print("="*60)
    
    mor_model = MoRViTModel(config).to(DEVICE)
    mor_optimizer = torch.optim.AdamW(mor_model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    mor_params = sum(p.numel() for p in mor_model.parameters())
    print(f"\nMoR-ViT Parameters: {mor_params:,}")
    
    mor_train_accs = []
    mor_test_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        # Assuming the new model does not return an auxiliary loss, so is_mor=False
        # If it does, you might need to adjust train_epoch or set is_mor=True
        train_loss, train_acc = train_epoch(mor_model, trainloader, mor_optimizer, classifier, DEVICE, is_mor=False)
        test_loss, test_acc = evaluate(mor_model, testloader, DEVICE, is_mor=False)
        
        mor_train_accs.append(train_acc)
        mor_test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(6, 5))
    plt.plot(mor_train_accs, label='MoR-ViT Train', marker='s')
    plt.plot(mor_test_accs, label='MoR-ViT Test', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('MoR-ViT Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mor_vit_new_model_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final MoR-ViT Test Accuracy: {mor_test_accs[-1]:.2f}%")
    print("="*60)
    
if __name__ == '__main__':
    main()
