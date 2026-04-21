# Parkvision: Parking Space Detection

A Streamlit-based web application that uses a custom YOLOv8 model (`best.pt`) to detect **empty** and **occupied** parking spaces in parking lot images.

---

## Project Structure

```text
<Group03>_<6688033_6688045>/│
├── app.py                    # Streamlit web application
├── best.pt                   # Trained YOLOv8 model weights
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker build instructions
├── .dockerignore             # Docker ignore rules
├── data_preparation.ipynb    # Data preprocessing notebook (Colab)
├── model_development.ipynb   # Model training notebook (Colab)
├── demo.mov 
├── class_distribution.png
├── docs/               
│   ├── Final_Report.pdf
│   └── Presentation.pdf
│── outputs/
│   ├── predictions/
│   │   └── test_predictions.png
│   ├── analysis/
│   │   └── overfitting_analysis.png
│   └── comparison/
│       ├── all_models_mAP_curves.png
│       ├── baseline_training_curves.png
│       ├── confusion_matrix_comparison.png
│       ├── iter2_comparison.png
│       ├── iter3_comparison.png
│       ├── iteration_progress.png
│       └── model_comparison.csv
│── final_model_results/               
│   ├── confusion_matrix_normalized.png
│   |── confusion_matrix.png
│   |── results.png
    |── results.csv
│
├── README.md               
```
---

## Dataset

The project uses the **PKLot** dataset (UFPR04, UFPR05, PUCPR parking lots), originally sourced from [Kaggle](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset).

**Download `data.zip` (pre-processed, ready to use):** [Google Drive Link](https://drive.google.com/file/d/1b34Kp5LC658G2bT7-m80mdUlc2gd62RI/view?usp=sharing)

Unzipping `data.zip` produces a `dataset_split/` folder already split into `train/`, `valid/`, and `test/` subfolders — **no additional data preparation is needed.**

```text
dataset_split/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

> If you want to re-run the preprocessing, see `data_preparation.ipynb` and download the raw PKLot dataset from Kaggle.

---

## How to Reproduce Results

### Step 1 — Prepare the Dataset

1. Download `data.zip` from the Google Drive link above.
2. Unzip it so that `dataset_split/` is located at `/content/data/dataset_split/` in Google Colab.
3. That's it — the dataset is ready. You can skip to Step 2.

> **Optional:** To reproduce the preprocessing pipeline yourself, open `data_preparation.ipynb` in Colab, download the raw PKLot dataset from [Kaggle](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset), place it at `/content/PKLot`, and run all cells in order.

### Step 2 — Model Development (Google Colab)

1. Open `model_development.ipynb` in Google Colab (GPU runtime recommended).
2. Make sure `dataset_split/` is placed at `/content/data/dataset_split/` (from Step 1).
3. Run all cells in order. The notebook will train **3 iterations**:

| Iteration | Model | Key Changes |
|-----------|-------|-------------|
| 1 | YOLOv8n Baseline | No augmentation, 15 epochs |
| 2 | YOLOv8n + Augmentation | `augment=True`, dropout, weight decay, 50 epochs |
| 3 | YOLOv8n + Aug (Tuned) | Lower LR (`lr0=0.003`), reduced Mosaic (`mosaic=0.5`) |

4. After training, the best model weights will be saved at:
   `/content/runs/detect/<run_name>/weights/best.pt`
5. Copy `best.pt` into your project folder to use with the web app.

### Step 3 — Run the Web App (Docker)

Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed, then follow the steps below.

#### Build the Docker Image

```bash
docker build -t parking-app .
```

> Don't forget the `.` at the end — it specifies the current directory.

#### Run the Docker Container

```bash
docker run -p 8501:8501 parking-app
```

#### Access the Web App

Open your browser and go to:

[http://localhost:8501](http://localhost:8501)

#### Stop the Application

Press `Ctrl + C` in the terminal.

---

## Requirements

All Python dependencies are listed in `requirements.txt`. When using Docker, these are installed automatically during the image build.

---

## Notes

- GPU is strongly recommended for model training (Google Colab with T4/A100).
- The final `best.pt` used in the app corresponds to **Iteration 3** (YOLOv8n + Aug Tuned), which achieved the best test-set performance.