# Cardiac SDF Model — Demo

Simple self-contained demo that runs neural SDF inference on a cardiac MRI segmentation and produces an interactive 3D mesh visualization.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

This will:
1. Load the pretrained SDF model
2. Load a patient segmentation (ACDC dataset, patient001 ED frame)
3. Extract endo/epi contours and run the neural network
4. Generate an interactive 3D plot saved as `output_mesh.html`

## Files

- `run_demo.py` — Main script (model + inference + plotting, all in one file)
- `model/` — Contains the pretrained checkpoint (`.ptrom`)
- `data/patient001/` — Sample ACDC patient data (MRI + segmentation)
- `requirements.txt` — Python dependencies
