# Capstone Project – EDA (Exploratory Data Analysis) & Baseline Model

**Author**: Banajit Goswami

**Summary of Findings**
- Integrated MovieLens 100K ratings with TMDB 5000 metadata to create a 100,000-row enriched dataset (user demographics, genres, budget, runtime, cast excerpts).
- Data cleaning addressed missing runtime, budget, and popularity fields; duplicate (user, movie, timestamp) combinations were absent.
- Exploratory analysis revealed:
  - Ratings skew toward 4–5 stars, reflecting positivity bias.
  - Long-tailed distributions for user activity and movie popularity, highlighting the importance of cold-start diagnostics.
  - Drama, Comedy, and Action dominate genre frequency; budget/runtime show weak linear correlation with ratings, motivating nonlinear models.
- Engineered user preference vectors, content multi-hot genres, and a genre match interaction score for modeling.
- Baseline cosine KNN regressor (k=35) achieved:
  - Overall RMSE ≈ **1.12** and MAE ≈ **0.88**
  - Warm-start RMSE ≈ **1.12**, MAE ≈ **0.88**
  - Cold-start RMSE ≈ **1.27**, MAE ≈ **1.00** (movies with <5 historical ratings)
- Residual analysis shows heavier tails on cold-start predictions, confirming the need for richer feature sets in later submissions.

**Repository Map**
- [`capstone_eda_baseline.ipynb`](capstone_eda_baseline.ipynb) – Full exploratory analysis, feature engineering, and KNN baseline workflow.
- [`src/`](src) – Shared Python modules for data loading, feature engineering, and modeling utilities.

**How to Reproduce**
1. Activate a Python environment with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `scipy`.
2. From this directory, run the notebook:  
   `jupyter notebook capstone_eda_baseline.ipynb`
3. Ensure the data folders remain at `./Data/ml-100k` and `./Data/TMDB-5000`.

**Next Steps Preview**
- Incorporate Random Forest and neural recommenders with hyperparameter tuning.
- Add TF-IDF embeddings for movie overviews to reduce cold-start error.
- Produce comparative evaluation plots across models for Submission 3.





