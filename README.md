# Vision-MoR
We have implemented a Mixture of Regressions (MoR) inspired architecture for vision tasks. For base model, we have used the ViT architecture from transformers library, with the configuration of pretrained model `google/vit-base-patch16-224`, and compared it against our model. This repository contains the code and instructions to reproduce the results presented below.

### Configuration
For current version, we have used a pretrained configuration of ViT model `google/vit-base-patch16-224`. The model has been fine-tuned on CIFAR-10 dataset for 3 epochs(due to computation constraints) with a batch size of 110. The training was performed on a single google colab free-tier GPU (NVIDIA Tesla T4).

## Installation
To install the required dependencies, please run the following command:

```bash
pip install -r requirements.txt
```
## Usage
To compare the MoR and against the ViT, use the following command: 

```bash
python -m scripts.custom_config.metrics
```

This will run both models on CIFAR-10 dataset and output the results for comparison.

## Results
The results of our experiment are as follows:

| Metric          | ViT architecture | Vision MoR (our) | Improvement |
|----------------|------------------|------------------|-------------|
| Parameters (M) | 86.39 | 36.78 | 57.4% |
| Model Size (MB) | 329.55 | 140.29 | 57.4% |
| Inference Latency (ms) | 1160.35 | 1155.34 | 0.4% |
| Throughput (samples/s) | 94.80 | 95.21 | 0.4% |
| GFLOPs | 16.85 | 15.45 | 8.3% |
| Final Test Accuracy (%) | 46.04 | 47.54 | 1.50% |
| Peak Training Memory (MB) | 13755.05 | 13486.61 | 2.0% |

![results](image.png)

## Conclusion
The Vision MoR architecture demonstrates similar Training and Test Accuracy and Loss compared to a classic ViT, while achieving significant improvement in Parameters and Model Size for CIFAR-10 dataset.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work was inspired by the paper [Mixture of Recursion (MoR)](https://arxiv.org/abs/2507.10524).