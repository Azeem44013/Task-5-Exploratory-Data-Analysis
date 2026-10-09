import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nbformat as nbf
from matplotlib.backends.backend_pdf import PdfPages

DATA_PATH = Path(r"c:\Users\acer\Downloads\Task-5-Exploratory-Data-Analysis\train.csv")
OUTPUT_DIR = Path(r"c:\Users\acer\Downloads\Task-5-Exploratory-Data-Analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def init_df():
    df = pd.read_csv(DATA_PATH)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# Titanic Exploratory Data Analysis

This notebook explores the Titanic dataset to identify patterns, trends, and anomalies related to survival.

## Goals
- Check dataset shape, missing values, and variable types
- Investigate univariate and bivariate relationships
- Highlight strong survival patterns by sex, class, and fare
- Summarize actionable insights from the visual analysis"""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv(r'c:\\Users\\acer\\Downloads\\Task-5-Exploratory-Data-Analysis\\train.csv')
train.head()"""
        ),
        nbf.v4.new_code_cell("""train.info()
train.isnull().sum()"""),
        nbf.v4.new_code_cell("""train.describe(include='all').T"""),
        nbf.v4.new_markdown_cell(
            """## Observation 1: Data quality and missing values
- There are 891 entries and 12 columns.
- Age has 177 missing values and Cabin has a large number of missing values, so it should be treated carefully in any model.
- Embarked has only 2 missing entries, so we can fill them with the mode."""
        ),
        nbf.v4.new_code_cell(
            """train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['HasCabin'] = train['Cabin'].notna().astype(int)

train[['Age', 'Embarked', 'FamilySize', 'HasCabin']].head()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Observation 2: Survival is strongly associated with sex and travel class
Women had a much higher survival rate than men, and first-class passengers survived more often than passengers in lower classes."""
        ),
        nbf.v4.new_code_cell(
            """survival_rate = train['Survived'].mean()
sex_survival = train.groupby('Sex')['Survived'].mean().sort_values(ascending=False)
class_survival = train.groupby('Pclass')['Survived'].mean().sort_values(ascending=False)
embarked_survival = train.groupby('Embarked')['Survived'].mean().sort_values(ascending=False)

print(f'Survival rate overall: {survival_rate:.2%}')
print("Survival by sex:\n", sex_survival)
print("Survival by class:\n", class_survival)
print("Survival by embarkation port:\n", embarked_survival)"""
        ),
        nbf.v4.new_code_cell(
            """sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x=sex_survival.index, y=sex_survival.values, ax=axes[0])
axes[0].set_title('Survival Rate by Sex')
axes[0].set_ylabel('Survival rate')

sns.barplot(x=class_survival.index, y=class_survival.values, ax=axes[1])
axes[1].set_title('Survival Rate by Pclass')
axes[1].set_ylabel('Survival rate')

sns.barplot(x=embarked_survival.index, y=embarked_survival.values, ax=axes[2])
axes[2].set_title('Survival Rate by Embarked Port')
axes[2].set_ylabel('Survival rate')

plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Observation 3: Age and fare distributions reveal wealth and age-related differences
Passengers who survived tended to be younger and had higher fares, which suggests a connection between passenger class and financial status."""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(train['Age'], bins=30, kde=True, ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Age Distribution')

sns.boxplot(x='Survived', y='Age', data=train, ax=axes[0,1], palette='Set2')
axes[0,1].set_title('Age by Survival')

sns.histplot(train['Fare'], bins=30, kde=True, ax=axes[1,0], color='darkorange')
axes[1,0].set_title('Fare Distribution')

sns.boxplot(x='Survived', y='Fare', data=train, ax=axes[1,1], palette='pastel')
axes[1,1].set_title('Fare by Survival')

plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Observation 4: Correlation and pairwise relationships
The strongest numeric relationships are found between passenger class, fare, and survival, while family size and Parch/SibSp show moderate association."""
        ),
        nbf.v4.new_code_cell(
            """numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
correlation = train[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', center=0, linewidths=.5, ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """pair_data = train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
sns.pairplot(pair_data, hue='Survived', palette='husl', diag_kind='hist', plot_kws={'alpha': 0.7})
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Final insights
1. Sex was the strongest factor for survival: women survived much more often than men.
2. Higher-class passengers had more access to lifeboats and better cabins, which translated to higher survival rates.
3. Fare was positively associated with survival, confirming the class effect.
4. Age was a secondary factor; younger passengers and children were more likely to survive than older adults.
5. Missing data in Age and Cabin should be handled carefully during modeling, but the overall dataset still clearly shows actionable patterns."""
        ),
    ]
    return nb


def make_analysis_plots(df: pd.DataFrame):
    sns.set_theme(style='whitegrid')

    sex_survival = df.groupby('Sex')['Survived'].mean().sort_values(ascending=False)
    class_survival = df.groupby('Pclass')['Survived'].mean().sort_values(ascending=False)
    port_survival = df.groupby('Embarked')['Survived'].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(x=sex_survival.index, y=sex_survival.values, ax=axes[0])
    axes[0].set_title('Survival Rate by Sex')
    axes[0].set_ylabel('Survival Rate')
    sns.barplot(x=class_survival.index, y=class_survival.values, ax=axes[1])
    axes[1].set_title('Survival Rate by Passenger Class')
    axes[1].set_ylabel('Survival Rate')
    sns.barplot(x=port_survival.index, y=port_survival.values, ax=axes[2])
    axes[2].set_title('Survival Rate by Embarkation Port')
    axes[2].set_ylabel('Survival Rate')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'survival_by_groups.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(df['Age'], bins=30, kde=True, ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Age Distribution')
    sns.boxplot(x='Survived', y='Age', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Age by Survival')
    sns.histplot(df['Fare'], bins=30, kde=True, ax=axes[1, 0], color='darkorange')
    axes[1, 0].set_title('Fare Distribution')
    sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Fare by Survival')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'age_fare_distributions.png', dpi=200)
    plt.close(fig)

    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0, linewidths=.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=200)
    plt.close(fig)

    pair_data = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
    pair_fig = sns.pairplot(pair_data, hue='Survived', palette='husl', diag_kind='hist', plot_kws={'alpha': 0.7})
    pair_fig.fig.savefig(PLOTS_DIR / 'pairplot.png', dpi=200)
    plt.close(pair_fig.fig)


def export_pdf_report(df: pd.DataFrame):
    plt.rcParams['font.family'] = ['Calibri', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.titleweight'] = 'bold'

    report_path = OUTPUT_DIR / 'titanic_eda_report.pdf'
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.08, 0.96, 'Titanic Exploratory Data Analysis', fontsize=36, fontweight='bold', family='Calibri')
        fig.text(0.08, 0.88, 'Exploratory Data Analysis Report', fontsize=24, family='Calibri')

        info_x, y = 0.08, 0.78
        fig.text(info_x, y, 'Objective:', fontsize=20, fontweight='bold', family='Calibri')
        fig.text(info_x, y - 0.06, 'Identify patterns and anomalies linked to survival.', fontsize=16, family='Calibri')
        fig.text(info_x, y - 0.15, f'Dataset size: {df.shape[0]} rows and {df.shape[1]} columns', fontsize=16, family='Calibri')
        fig.text(info_x, y - 0.24, f'Survival rate: {df["Survived"].mean():.2%}', fontsize=16, family='Calibri')

        fig.text(info_x, 0.53, 'Key findings:', fontsize=20, fontweight='bold', family='Calibri')
        key_points = [
            '- Women had much higher survival than men',
            '- First-class passengers survived more often',
            '- Higher fares corresponded to better survival outcomes',
            '- Age and family size showed secondary influence',
        ]
        for idx, point in enumerate(key_points):
            fig.text(info_x, 0.47 - idx * 0.06, point, fontsize=16, family='Calibri')

        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.08, 0.96, 'Survival Patterns by Group', fontsize=24, fontweight='bold', family='Calibri')
        img = plt.imread(PLOTS_DIR / 'survival_by_groups.png')
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.68])
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.08, 0.08, 'Interpretation: Sex and passenger class are the clearest signals in the data.', fontsize=16, family='Calibri')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.08, 0.96, 'Distribution and Risk Signals', fontsize=24, fontweight='bold', family='Calibri')
        img = plt.imread(PLOTS_DIR / 'age_fare_distributions.png')
        ax = fig.add_axes([0.06, 0.20, 0.88, 0.62])
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.08, 0.08, 'Interpretation: Age and fare distributions show that wealth and cabin class greatly influenced survival odds.', fontsize=16, family='Calibri')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.08, 0.96, 'Correlation and Pairwise Relationships', fontsize=24, fontweight='bold', family='Calibri')
        img = plt.imread(PLOTS_DIR / 'correlation_heatmap.png')
        corr_ax = fig.add_axes([0.12, 0.50, 0.76, 0.28])
        corr_ax.imshow(img)
        corr_ax.axis('off')
        fig.text(0.08, 0.38, 'Summary: Pclass, Fare, and Sex show the strongest relations to survival, while age and family-size effects are moderate.', fontsize=16, family='Calibri')
        pair_img = plt.imread(PLOTS_DIR / 'pairplot.png')
        pair_ax = fig.add_axes([0.12, 0.10, 0.76, 0.24])
        pair_ax.imshow(pair_img)
        pair_ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


def main():
    df = init_df()
    make_analysis_plots(df)
    export_pdf_report(df)
    nb = create_notebook()
    with open(OUTPUT_DIR / 'titanic_eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Created report at {OUTPUT_DIR / "titanic_eda_report.pdf"}')
    print(f'Created notebook at {OUTPUT_DIR / "titanic_eda.ipynb"}')


if __name__ == '__main__':
    main()
