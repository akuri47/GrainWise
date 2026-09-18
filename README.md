# 🌾 GrainWise — Rice Grain Quality Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://grainwise.streamlit.app)

**Computer Vision + Data Analytics** system for automated rice quality assessment, variety identification, shelf-life estimation, and market price recommendation — built per Government of India FAQ standards.
 
> 🌐 **Live Analytics Dashboard:** [https://grainwise.streamlit.app](https://grainwise.streamlit.app)

---

## 📋 Overview

GrainWise analyses rice grain images to deliver a comprehensive quality report:

1. **Grain Segmentation** — ArUCo-calibrated CV pipeline segments individual grains from plate images  
2. **Defect Classification** — ResNet-18 CNN classifies grains into 6 categories (whole, broken, chalky, damaged, discolored, foreign)  
3. **Variety Identification** — Random Forest classifier identifies 9 Indian rice varieties using 11 morphological features  
4. **Quality Grading** — Grades samples as Grade A / B / Common / Rejected per GoI FAQ thresholds  
5. **Shelf-Life Estimation** — IRRI halving-rule model estimates storage life based on temperature, moisture, and defects  
6. **Price Recommendation** — Live Agmarknet API integration for market-aware pricing with quality-score adjustments  

---

## 🏗️ Project Structure

```
GrainWise/
├── src/                          # Core ML & analysis modules
│   ├── analyzer.py               # End-to-end analysis pipeline
│   ├── grain_segmentation.py     # ArUCo calibration + grain segmentation
│   ├── defect_classifier.py      # ResNet-18 defect CNN
│   ├── variety_classifier.py     # Random Forest variety classifier
│   ├── quality_assessment.py     # FAQ grading & quality scoring
│   ├── shelf_life.py             # Shelf-life estimation (halving rule)
│   ├── price_recommendation.py   # Pricing engine + deductions
│   ├── mandi_api.py              # Live Agmarknet API integration
│   └── config.py                 # Central configuration
│
├── data_analytics/               # 📊 Data Analytics Module
│   ├── eda_analysis.py           # EDA — heatmaps, box plots, distributions
│   ├── statistical_tests.py      # ANOVA, chi-square, Spearman correlation
│   ├── streamlit_dashboard.py    # Interactive Streamlit dashboard
│   ├── generate_sample_data.py   # Sample data generator
│   └── requirements.txt          # Analytics dependencies
│
├── web/                          # Flask backend (REST API)
│   └── app.py                    # WSGI app served via Gunicorn
│
├── vercel_frontend/              # PWA frontend (hosted on Vercel)
│
├── models/                       # Trained model weights
│   ├── defect_resnet18_best.pth  # ResNet-18 defect classifier
│   └── variety_classifier.joblib # Random Forest variety model
│
└── requirements_prod.txt         # Production dependencies
```

---

## 🔬 Core ML Pipeline

### Defect Classifier (ResNet-18)
| Metric | Value |
|--------|-------|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Training Data | 2,200 grain images (6 classes) |
| Accuracy | 86.4% |
| Macro F1 | 0.86 |
| Augmentation | Flips, rotation, color jitter + offline minority oversampling |

### Variety Classifier (Random Forest)
| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (200 trees) |
| Training Data | 1,064 grains (9 varieties) |
| Accuracy | 79.5% (5-fold stratified CV) |
| Balancing | SMOTE + majority downsampling |
| Features | 11 morphological + color features |

---

## 📊 Data Analytics Module

The `data_analytics/` folder contains standalone analytics tools for exploring grain quality data.

### 1. Exploratory Data Analysis
```bash
python data_analytics/eda_analysis.py
```
Generates correlation heatmaps, morphological box plots, defect distribution charts, and variety-wise scatter plots → saved to `data_analytics/plots/`.

### 2. Statistical Hypothesis Testing
```bash
python data_analytics/statistical_tests.py
```
Runs ANOVA, Kruskal-Wallis, chi-square, and Spearman correlation tests → confirms significant inter-variety morphological differences (p < 0.001).

### 3. Interactive Dashboard

> 🔗 **Live Demo:** [https://grainwise.streamlit.app](https://grainwise.streamlit.app)

```bash
pip install -r data_analytics/requirements.txt
streamlit run data_analytics/streamlit_dashboard.py
```
Launches a Streamlit dashboard with:
- **KPI Cards** — Total grains, varieties, quality score, whole grain %
- **Interactive Charts** — Defect pie, variety bar, feature box plots, scatter plots
- **Filters** — By variety, defect type, quality grade
- **Data Export** — Download filtered grain data as CSV

### 4. Sample Data Generation
```bash
python data_analytics/generate_sample_data.py
```
Generates realistic synthetic grain feature data for analytics testing.

---

## 🌐 Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| Analytics Dashboard | Streamlit Community Cloud | [https://grainwise.streamlit.app](https://grainwise.streamlit.app) |
| Backend API | Render (Flask + Gunicorn) | REST API endpoint |
| Frontend PWA | Vercel CDN | Progressive Web App |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Computer Vision** | OpenCV, NumPy, ArUCo markers |
| **Deep Learning** | PyTorch, ResNet-18, Transfer Learning |
| **Machine Learning** | scikit-learn, Random Forest, SMOTE |
| **Data Analytics** | Pandas, Seaborn, Matplotlib, SciPy, Plotly |
| **Dashboard** | Streamlit |
| **API Integration** | Agmarknet (data.gov.in) live mandi prices |
| **Backend** | Flask, Gunicorn (WSGI) |
| **Frontend** | HTML/CSS/JS (PWA), Vercel |
| **Standards** | Government of India FAQ grading system |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/akuri47/grainwise.git
cd grainwise

# Install dependencies
pip install -r requirements_prod.txt

# Run the analytics dashboard
pip install -r data_analytics/requirements.txt
streamlit run data_analytics/streamlit_dashboard.py

# Run the web app locally
python web/app.py
```

---

## 📄 License

This project was developed as part of a research internship at IIT Kharagpur under the supervision of Prof. Brajesh K. Panda.
