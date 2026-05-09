# IBM HR Analytics - Employee Attrition EDA

**Author:** Jimmy Le-Nguyen  
**GitHub:** [github.com/jimmyle9080](https://github.com/jimmyle9080)

## Overview

An end-to-end exploratory data analysis and attrition prediction project built on the IBM HR Analytics Employee Attrition dataset. The dataset contains **1,470 employee records across 35 features**, with a **16.1% attrition rate** -- a realistic workforce analytics problem directly applicable to HR strategy, consulting, and business analytics roles.

This project covers the full analytics lifecycle: data cleaning and feature engineering, SQL analysis, business insight generation, machine learning model development, and Power BI-ready reporting exports.

## Key Business Findings

| Finding | Detail |
|---|---|
| Overall attrition rate | 16.1% (237 of 1,470 employees) |
| Highest attrition department | Sales |
| Highest attrition job role | Sales Representative |
| Overtime attrition rate | ~31% vs ~10% for non-overtime |
| Avg income (left) | Lower than employees who stayed |
| High risk employees flagged | Low satisfaction + overtime |

## Dataset

- **Source:** [Kaggle - IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **1,470 employees** | **35 features** | **16.1% attrition rate**
- Covers demographics, job details, satisfaction scores, compensation, and tenure

> Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from Kaggle and place it in the `data/` folder before running.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (VS Code: just press the play button on main.py)
python main.py
```

Everything runs from `main.py` — no additional setup needed.

## Project Structure

```
ibm_hr_project/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Kaggle dataset
├── outputs/
│   ├── charts/      # 6 professional visualizations
│   └── exports/     # Power BI-ready CSV exports
├── main.py          # Single entry point — run this
├── requirements.txt
└── README.md
```

## Outputs

### Charts (outputs/charts/)
| File | Description |
|---|---|
| 1_attrition_overview.png | Overall rate, by department, by age group |
| 2_jobrole_analysis.png | Attrition rate and income by job role |
| 3_attrition_drivers.png | Overtime and job satisfaction impact |
| 4_income_tenure_analysis.png | Income distribution and tenure breakdown |
| 5_feature_importance.png | Top 15 attrition predictors (Random Forest) |
| 6_model_comparison.png | AUC-ROC, AUPRC, Recall across 3 models |

### Power BI Exports (outputs/exports/)
| File | Description |
|---|---|
| scored_employees.csv | All 1,470 employees with attrition risk scores |
| attrition_by_department.csv | Department-level attrition analysis |
| attrition_by_jobrole.csv | Role-level attrition and income breakdown |
| attrition_by_age_group.csv | Age group attrition patterns |
| overtime_impact.csv | Overtime vs non-overtime attrition comparison |
| satisfaction_vs_attrition.csv | Satisfaction score impact on attrition |
| tenure_attrition.csv | Attrition by years at company |
| high_risk_employees.csv | Flagged employees: low satisfaction + overtime |
| feature_importance.csv | Top predictive features from Random Forest |
| model_comparison.csv | Full model performance metrics |

## Technical Approach

### Feature Engineering
- `TenureGroup` -- grouped years at company into career stage buckets
- `AgeGroup` -- generational cohort grouping
- `SatisfactionScore` -- composite of Job, Environment, Relationship, WorkLife satisfaction
- `AttritionFlag` -- binary target variable (0/1)
- `IncomePerYear` -- annualized monthly income

### SQL Analysis
SQLite used for all business-level aggregations:
- Attrition rates by department, job role, age group, and tenure
- Overtime impact analysis
- Satisfaction score breakdown
- High-risk employee identification (low satisfaction + overtime)

### Models
Three models trained and benchmarked on attrition prediction:
1. **Logistic Regression** -- interpretable baseline with class weighting
2. **Random Forest** -- ensemble method, strong feature importance output
3. **Gradient Boosting** -- sequential ensemble for high recall on attrition class

Final risk scoring uses an **ensemble probability** (average of RF + GB) for employee risk tiering: Low / Medium / High / Critical.

### Class Imbalance Handling
The dataset has a roughly 5:1 imbalance (stayed:left). Addressed through:
- **Class weighting** in Logistic Regression and Random Forest
- **AUPRC as supplementary metric** alongside AUC-ROC
- **Stratified train/test split** to preserve attrition ratio

## Skills Demonstrated

- **Python** -- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **SQL** -- SQLite queries with aggregations, filtering, and business logic
- **EDA** -- Univariate and multivariate analysis across 35 features
- **Machine Learning** -- Classification, ensemble methods, class imbalance handling
- **Feature Engineering** -- Derived features, composite scores, categorical grouping
- **Model Evaluation** -- AUC-ROC, AUPRC, Precision, Recall, F1
- **Business Storytelling** -- Translating analytical findings into HR strategy recommendations
- **Power BI Integration** -- Structured CSV exports for live dashboard connection

## License

Dataset licensed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).  
Code: MIT License.
