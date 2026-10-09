\# Salary Prediction App



An end-to-end machine learning project: a regression model that predicts \*\*salary from years of experience\*\*, served through an interactive \*\*Streamlit\*\* web app.



\*\* Live app:\*\* \[https://salary-predictor-app-demo.streamlit.app](https://salary-predictor-app-demo.streamlit.app)



!\[App prediction](screenshots/app-prediction.png)

!\[Prediction history](screenshots/app-prediction-history.png)

!\[Dataset tab](screenshots/app-dataset.png)

!\[Model performance tab](screenshots/app-model-performance.png)



\## Project description

The app lets a user enter their years of experience and returns a predicted salary. It also shows the dataset, the model's performance metrics, a chart of the prediction on the fitted line, and a history of predictions made in the session. The model is trained once in a notebook, saved with `joblib`, and \*\*loaded (not retrained)\*\* by the app.



\## Dataset

\- \*\*File:\*\* `data/salary\_dataset.csv`, 200 records, 2 columns (`Experience Years`, `Salary`).

\- \*\*Source:\*\* Salary dataset provided in class (GitHub).

\- \*\*Quality:\*\* no missing values or duplicates. The salary currency is not specified in the dataset, so values are shown in the dataset's own units.



\## Model

\- \*\*Algorithm:\*\* Linear Regression (scikit-learn). Polynomial models (degree 2 and 3) were compared and gave no meaningful improvement, so the simpler, more interpretable model was kept.

\- \*\*Features / target:\*\* `Experience Years` → `Salary`.

\- \*\*Validation:\*\* 80/20 train/test split (`random\_state=42`) plus 5-fold \*\*shuffled\*\* cross-validation. The file is sorted by experience, so unshuffled folds would give misleading scores.

\- \*\*Final model:\*\* refit on all 200 rows and saved to `model/model.pkl`.



\## Evaluation metrics

| Metric | Value |

|---|---|

| R² (test set) | 0.983 |

| MAE (test set) | 4,056 |

| RMSE (test set) | 4,873 |

| R² (5-fold shuffled CV, mean ± SD) | 0.980 ± 0.005 |



Fitted equation: `salary ≈ 25,641 + 9,469 × years of experience`.



\*\*Limitation:\*\* experience is the only feature, and predictions are reliable only within the training range (0.6 to 14.8 years). The app warns when the input is outside that range.



\## Project structure

```

ML\_Project/

├── data/salary\_dataset.csv

├── model/

│   ├── model.pkl            # serialized model (joblib)

│   └── metrics.json         # evaluation metrics shown in the app

├── notebooks/model\_training.ipynb

├── screenshots/             # screenshots of the deployed app

├── app.py                   # Streamlit application

├── requirements.txt

├── README.md

└── .gitignore

```



\## Run locally

Python 3.11 or newer is required.



```bash

git clone https://github.com/naimasiddiqui10/salary-prediction-app.git

cd salary-prediction-app



python -m venv venv

\# Windows:      venv\\Scripts\\activate

\# macOS/Linux:  source venv/bin/activate



pip install -r requirements.txt

streamlit run app.py

```



To retrain the model: `pip install notebook`, open `notebooks/model\_training.ipynb` and run all cells. This rewrites `model/model.pkl` and `model/metrics.json`.



\## Tech stack

Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Altair, Matplotlib, Seaborn.

