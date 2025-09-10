[![Zero-Shot Image Editing](https://img.shields.io/badge/zero%20shot-image%20editing-Green)]([https://github.com/topics/video-editing](https://github.com/topics/text-guided-image-editing))
[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![torch](https://img.shields.io/badge/torch-2.0.0+-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-yellow?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)


# FlowOpt: Fast Optimization Through Whole Flow Processes For Training-Free Editing

![Teaser](assets/teaser.jpg)

**This is the anonymous submission version. The final version will be published after acceptance (and after code-cleanup).**

## Getting Started
### 1. Clone the repo

```bash
git clone https://github.com/orronai/FlowOpt.git
cd FlowOpt
```

### 2. Install the required environment

```bash
python -m pip install -r requirements.txt
```

### 3. Usage Example
#### A. Inversion

For FLUX:

```bash
python FlowOpt.py --exp_yaml yaml_files/FLUX_inversion_exp.yaml
```

For SD3:

```bash
python FlowOpt.py --exp_yaml yaml_files/SD3_inversion_exp.yaml
```

#### B. Editing

For FLUX:

```bash
python FlowOpt.py --exp_yaml yaml_files/FLUX_editing_exp.yaml
```

FOR SD3:
```bash
python FlowOpt.py --exp_yaml yaml_files/SD3_editing_exp.yaml
```
