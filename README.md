# Customer Segmentation

Customer clustering pipeline for the UCI Online Retail dataset. The workflow cleans transactions, engineers customer-level features, explores PCA, and fits K-Means segments with visual diagnostics (2D/3D plots, radar charts) plus SHAP-based interpretability for the surrogate classifier.

## Repository structure
- data/
	- raw/online_retail.csv — source dataset.
	- processed/ — cleaned data, engineered features, and saved cluster assignments (k=3, k=4).
- notebooks/
	- 01_clean_eda.ipynb — load, clean UK subset, quick EDA; saves cleaned_uk_data.csv.
	- 02_feature_engineering.ipynb — build customer features, Box-Cox + scaling; saves feature CSVs.
	- 03_modeling.ipynb — PCA, optimal k search, K-Means (k=3/4), radar plots, surrogate RandomForest + SHAP.
- src/
	- clustering_library.py — classes for cleaning (DataCleaner), feature engineering (FeatureEngineer), clustering + plots + SHAP (ClusterAnalyzer).
- requirements.txt — Python dependencies.

## Getting started
1) Create a virtual environment (recommended):
```
python -m venv .venv
.\.venv\Scripts\activate
```
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Ensure data/raw/online_retail.csv is present (already provided here).

## Typical workflow (notebooks)
1. Run 01_clean_eda.ipynb to clean the UK subset and inspect distributions; output: data/processed/cleaned_uk_data.csv.
2. Run 02_feature_engineering.ipynb to aggregate per-customer features, apply Box-Cox, and standardize; outputs: data/processed/customer_features*.csv.
3. Run 03_modeling.ipynb to:
	 - Apply PCA and inspect variance.
	 - Find optimal k (elbow + silhouette).
	 - Fit K-Means for k=3 and k=4; visualize in 2D/3D and radar charts.
	 - Train RandomForest surrogates and compute SHAP values to explain cluster boundaries.
	 - Save cluster assignments to data/processed/customer_clusters_k3.csv and customer_clusters_k4.csv.

## Key components
- DataCleaner: loads raw CSV, removes cancellations/negatives, keeps UK, adds TotalPrice, computes RFM, and writes cleaned_uk_data.csv.
- FeatureEngineer: builds 16 customer-level features, Box-Cox transforms, scales, and writes feature CSVs.
- ClusterAnalyzer: PCA, k search, K-Means, plots (2D/3D, radar), surrogate RandomForest, and SHAP summaries.

## Notes
- SHAP and RandomForest steps can be compute-intensive; consider sampling for quick iterations.
- Figures save inline in notebooks; ensure a graphical backend is available when running locally.