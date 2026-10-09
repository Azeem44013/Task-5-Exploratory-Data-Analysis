# Titanic Exploratory Data Analysis

This project performs exploratory data analysis on the Titanic dataset to identify survival patterns, trends, and anomalies.

## Objective
Use Python, Pandas, Matplotlib, and Seaborn to explore the dataset and generate a notebook plus a PDF report summarizing the findings.

## Dataset
The analysis uses the Kaggle Titanic training dataset available in this folder as `train.csv`.

## Deliverables
- `titanic_eda.ipynb` — Jupyter notebook containing the full EDA workflow
- `titanic_eda_report.pdf` — PDF report of the key findings and major visuals
- `build_titanic_eda.py` — script to regenerate the notebook and PDF from the dataset
- `plots/` — saved charts used in the analysis and report

## Key findings
- Women had a much higher survival rate than men.
- First-class passengers survived substantially more often than passengers in lower classes.
- Higher fares were strongly related to better survival outcomes.
- Age distributions showed that younger passengers were somewhat more likely to survive.
- Missing values in `Age` and `Cabin` should be treated carefully in modeling tasks.

## Run it locally
```bash
python build_titanic_eda.py
```

This recreates the notebook and the PDF report in the project directory.
