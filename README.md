# 🛰️ Multimodal Property Valuation

### *Fusing Satellite Imagery with Data for Real Estate Prediction*

## 📖 Overview
A **Multimodal Machine Learning Pipeline** predicting property values by combining traditional tabular data (e.g., bedrooms, sqft) with **Satellite Imagery** (neighborhood context like greenery and density). Using a **Late Fusion** architecture, it merges numerical features with visual embeddings (from **ResNet18**) to improve predictive accuracy.

## 🏗️ Architecture
1.  **Visual Branch:** Satellite images (Mapbox API) → ResNet18 (512-dim embedding) → PCA (20 components).
2.  **Tabular Branch:** King County dataset → Log-transform, Scaling, & Engineering.
3.  **Fusion:** Concatenation → **XGBoost Regressor** for final price prediction.

## 📂 Structure
* `data/`: Train, test, and processed datasets.
* `data_fetcher.py`: Mapbox API image downloader.
* `preprocessing.ipynb`: EDA, cleaning, and feature engineering.
* `model_training.ipynb`: CNN extraction, fusion, XGBoost training.

## 🚀 Usage
1.  **Install:**
    ```bash
    git clone https://github.com/bharataggarwal56/multimodal-property-valuation.git
    cd multimodal-property-valuation
    pip install pandas numpy scikit-learn xgboost torch torchvision pillow requests tqdm
    echo "MAPBOX_ACCESS_TOKEN=your_token_here" > .env
    ```
2.  **Run:**
    * `python data_fetcher.py` (Download images)
    * Run `preprocessing.ipynb` (Clean data)
    * Run `model_training.ipynb` (Train & Predict)

## 📊 Results
| Model | RMSE (Log) | R² Score |
| :--- | :--- | :--- |
| **Hybrid Fusion** | **0.1671** | **0.8980** |
| Tabular Baseline | 0.1715 | 0.8926 |
| Image-Only | 0.4303 | 0.3241 |

## 🛠️ Tech Stack
**Python 3.8+**, **PyTorch** (ResNet18), **XGBoost**, **Pandas**, **Mapbox API**.
