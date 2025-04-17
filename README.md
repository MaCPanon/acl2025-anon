# MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection

This repository provides the implementation of **MaCP**, a lightweight adaptation method that leverages **Hierarchical Cosine Projection** in the discrete cosine domain. MaCP achieves high accuracy while significantly reducing training cost and GPU memory.

> 🔒 This repository is anonymized and intended for review purposes only.  
> 🔗 For anonymous access: [anonymous.4open.science link will go here]

---

## 📦 Installation

To set up the environment:

```bash
cd ./peft
pip install -e .
pip install datasets scipy scikit-learn evaluate pillow torchvision optuna wandb
```

We recommend using a Python environment manager like Conda or venv.

---

## 🚀 Running Experiments

All benchmark scripts are located in the `./experiments` folder. Each task folder contains example shell scripts with tuned hyperparameters.

### Example: GLUE Benchmark (Natural Language Understanding)

To run CoLA using RoBERTa-Large:

```bash
cd experiments/glue
bash cola-large-best.sh
```

For Slurm-based systems, make sure to:
- Load required modules (`anaconda`, `CUDA`, etc.)
- Specify GPU partition (`--gres=gpu:1`, etc.)

Scripts for other GLUE tasks (e.g., QNLI, RTE, SST-2) are similarly named.

---

## 📊 Reproducibility & Seeds

All reported results use the **median of 5 seeds**:

```
0, 11111, 22222, 33333, 44444
```

If a shared entry is used, the seed is unified to:

```
2025
```

> Due to variations in GPU type, PyTorch version, and driver stacks, some fluctuations may occur. If so, adjust hyperparameters slightly (e.g., `head_lr`, `macp_lr`, `scale`).

---

## 💻 GPU Memory Cost

To evaluate MaCP’s GPU usage compared to other methods:

```bash
bash eval-gpu-cost.sh
```

**Peak CUDA Memory (GB) on RoBERTa-Large**  
*(Batch size: 128 for CoLA, 32 for others)*

| Method    | SST-2 | MRPC | CoLA | QNLI | RTE  | STS-B |
|-----------|-------|------|------|------|------|-------|
| Full FT   | 14.3  | 13.7 | 20.6 | 19.1 | 16.7 | 14.1  |
| DoRA      | 10.8  | 8.2  | 14.1 | 13.6 | 10.9 | 9.0   |
| AFLoRA    | 10.0  | 7.8  | 12.8 | 12.3 | 10.4 | 8.5   |
| LaMDA     | 8.5   | 6.7  | 11.2 | 10.5 | 8.9  | 7.3   |
| **MaCP**  | **7.2** | **5.2** | **9.4** | **8.9** | **7.1** | **5.2** |

---

## 📁 Project Structure

```
MaCP/
│
├── peft/                    # Core MaCP & PEFT implementation
├── experiments/             # Task-specific scripts
│   ├── glue/                # GLUE benchmark (e.g., cola-large-best.sh)
│   ├── instruction_tuning/  # Instruction_tuning with MT-Bench and Vicuna 
│   └── vision/              # Image classification tasks with ViT
├── scripts/               # Utility scripts (e.g., eval-gpu-cost.sh)
├── README.md              # This file
└── requirements.txt       # Dependencies (optional)
```

---

## 📄 Highlights of MaCP

- **Spectral Efficiency**: Uses Discrete Cosine Transform (DCT) for weight updates
- **Hierarchical Sampling**: Low, mid, and high frequencies selected via distance-based partitioning
- **Memory-Aware**: Lower memory usage vs. LoRA and FourierFT
- **Versatile**: Validated across NLU, NLG, instruction tuning, vision, and video-language tasks

---

## 📌 Notes

- This codebase is prepared **anonymously** for the ACL 2025 review process.
- Authors will release full attribution and citation details **after acceptance**.

---

## 📜 Citation

> The citation will be available upon paper acceptance.  
> For now, please cite this repository as:  
> `"MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection", Anonymous Submission, ACL 2025.`

---

