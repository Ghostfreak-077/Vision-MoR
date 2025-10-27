# Vision-MoR
We have implemented a Mixture of Regressions (MoR) inspired architecture for vision tasks. For now, we have simply compared a basic architecture against a simple transformer. This repository contains the code and instructions to reproduce the results presented below.

## Installation
To install the required dependencies, please run the following command:

```bash
pip install -r requirements.txt
```
## Usage
To compare the MoR and against a simple Transformer, use the following command: 

```bash
python -m scripts.main
```

This will run both models on CIFAR-10 dataset and output the results for comparison.

## Results
The results of our experiments are as follows:

| Metric          | Classic Transformer | Vision MoR (our) | Improvement      |
|----------------|--------------|------------------|------------------|
| Parameters (M)       | 4.77        | 2.40            | 49.6%           |
| Model Size (MB)           | 18.20        | 9.17            | 49.6%           |
| Inference Latency (ms)   | 140.62       | 104.07          | 26.0%           |
| Throughput (samples/s)  | 3641.11      | 4919.69        | 35.1%           |
| GFLOPs                 | 0.31         | 0.20           | 33.5%           |
| Final Test Accuracy (%)   | 66.59        | 67.28          | 0.69%           |
| Peak Training Memory (MB) | 4001.23      | 2899.86        | 27.5%           |

![results](image.png)

## Conclusion
The Vision MoR architecture demonstrates similar Training and Test Accuracy and Loss compared to a classic transformer, while achieving significant improvement in Training time, throughput and efficiency in CIFAR-10 dataset.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work was inspired by the paper [Mixture of Recursion (MoR)](https://arxiv.org/abs/2507.10524).