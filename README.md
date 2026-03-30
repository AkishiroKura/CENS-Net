# CENS-Net

This repository contains the implementation of the model proposed in our paper **"CENS-Net: CENS-Net: Interpretable Edge-Node Learningfor Neurodegenerative Diseases Classification from White Matter Tractography"** using PyTorch and PyTorch Geometric.

## Authors and Contributors

This code was implemented by:

- [Your Name] (Email: [your.email@institution.edu])
- [Co-author Name] (Email: [co-author@institution.edu])
- [Other Contributors]

From [Your Institution/Department].

## Project Overview

CENS-Net is a graph neural network (GNN) model specifically designed for interpretable classification of neurodegenerative diseases, namely Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD), using white matter tractography data. The model leverages connectome edge and node statistics to achieve both high classification performance and explainability through gradient-based feature attribution methods.

### Key Features

- **Dual-path architecture** for capturing both edge-level and node-level connectivity patterns
- **Explainability-focused** design with integrated Gradient-based SHAP for feature importance analysis
- **White matter connectivity analysis** from diffusion MRI tractography data
- **Publication-ready visualizations** including CAM heatmaps, SHAP importance plots, and Circos-style chord diagrams

## Usage Notice

- Feel free to download and use this code for testing your algorithms and reproducing results.
- If you use this code in your publications, please cite our paper and inform us in advance.
- For questions or collaboration inquiries, please contact the corresponding authors.

Thank you for your cooperation!

Date: March 24, 2026

## Dependencies

This project is implemented in Python using the following major packages:

- **PyTorch** (≥1.9.0) – Deep learning framework for model implementation
- **PyTorch Geometric** (≥2.0.0) – Graph neural network library
- **NumPy & SciPy** – Numerical computing and scientific functions
- **Pandas** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Visualization
- **scikit-learn** – Machine learning utilities and metrics
- **SHAP** – For gradient-based feature attribution and explainability

### Installation

Create a Python environment and install dependencies:

```bash
# Create virtual environment
conda create -n cens-net python=3.9
conda activate cens-net

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy pandas matplotlib seaborn scikit-learn shap scipy
```

## Usage

### Training

```bash
# Train the model
python train.py --fold 1 --model dualpath

# Train with multiple folds (5-fold cross-validation)
for fold in {1..5}; do
    python train.py --fold $fold --model dualpath
done
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --fold 1 --model dualpath

# Generate comparison metrics
python compare_metrics.py
```

### Explainability Analysis

```bash
# Generate SHAP feature importance analysis
python run_explainability.py --fold 1

# Generate CAM heatmaps
python run_cam_heatmap.py --fold 1

# Generate Circos chord diagrams
python plot_chord_diagram_circos.py --all --top_k 20
```

## Datasets

The white matter tractography data were collected from:

- **Alzheimer's Disease (AD) patients** 
- **Frontotemporal Dementia (FTD) patients** 

Data acquisition and preprocessing protocols are described in the main paper.

### Data Availability




## Results

The CENS-Net model achieves competitive classification performance on AD vs. FTD discrimination with high explainability:

- **Classification Accuracy**: [Your metric here]
- **Sensitivity/Specificity**: [Your metrics here]
- **Key discriminative features**: Generalized Fractional Anisotropy (GFA) and fiber tract count

Detailed results including SHAP feature importance and per-metric analysis are provided in `results_explainability/`.

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{CENS-Net2026,
  title={CENS-Net: Leveraging Connectome Edge-Node Statistics for Interpretable AD and FTD Classification from White Matter Tractography},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI]}
}
```

## License

This project is provided as-is for research purposes. Please contact the authors for licensing inquiries.

## Acknowledgments

We acknowledge the support from [Institution/Funding Source] and all participants in this study.

---

For questions, suggestions, or technical issues, please open an issue on GitHub or contact the corresponding authors directly.
