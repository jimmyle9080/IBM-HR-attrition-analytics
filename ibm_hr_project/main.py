"""
IBM HR Analytics - Employee Attrition EDA
Author: Jimmy Le-Nguyen
GitHub: https://github.com/jimmyle9080

Dataset: IBM HR Analytics Employee Attrition (Kaggle)
1,470 employees | 35 features | 16.1% attrition rate

How to run:
    1. pip install -r requirements.txt
    2. Open folder in VS Code
    3. Run main.py (press the play button)

Outputs:
    - outputs/charts/  : 6 professional visualizations
    - outputs/exports/ : Power BI-ready CSV exports
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ── Colors ─────────────────────────────────────────────────────────────────────
ATTRITION_COLOR = "#E63946"
RETAIN_COLOR    = "#457B9D"
ACCENT_COLOR    = "#2A9D8F"
WARN_COLOR      = "#E9C46A"
BG_COLOR        = "#F8F9FA"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# ── Step 1: Load & Clean ───────────────────────────────────────────────────────
print("=" * 60)
print("  IBM HR ANALYTICS - EMPLOYEE ATTRITION EDA")
print("  Author: Jimmy Le-Nguyen")
print("=" * 60)

print("\n[1/6] Loading and cleaning data...")
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
df['AttritionFlag'] = (df['Attrition'] == 'Yes').astype(int)
df['OverTimeFlag']  = (df['OverTime'] == 'Yes').astype(int)
df['IncomePerYear']     = (df['MonthlyIncome'] * 12).round(0)
df['TenureGroup']       = pd.cut(df['YearsAtCompany'],
                                  bins=[-1, 2, 5, 10, 20, 100],
                                  labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs'])
df['AgeGroup']          = pd.cut(df['Age'],
                                  bins=[17, 25, 35, 45, 55, 100],
                                  labels=['18-25', '26-35', '36-45', '46-55', '55+'])
df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                            df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4

total     = len(df)
attrition = df['AttritionFlag'].sum()
retain    = total - attrition
attr_rate = attrition / total * 100

print(f"   Employees loaded     : {total:,}")
print(f"   Left company         : {attrition:,}  ({attr_rate:.1f}%)")
print(f"   Stayed               : {retain:,}  ({100-attr_rate:.1f}%)")
print(f"   Null values          : {df.isnull().sum().sum()}")

# ── Step 2: SQL Analysis ───────────────────────────────────────────────────────
print("\n[2/6] Running SQL analysis...")
conn = sqlite3.connect(":memory:")
df.to_sql("employees", conn, if_exists="replace", index=False)

queries = {
    "attrition_by_department": """
        SELECT Department,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
               ROUND(AVG(YearsAtCompany), 1) AS avg_tenure_years
        FROM employees
        GROUP BY Department
        ORDER BY attrition_rate_pct DESC
    """,
    "attrition_by_jobrole": """
        SELECT JobRole,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
        FROM employees
        GROUP BY JobRole
        ORDER BY attrition_rate_pct DESC
    """,
    "attrition_by_age_group": """
        SELECT AgeGroup,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
        FROM employees
        GROUP BY AgeGroup
        ORDER BY AgeGroup
    """,
    "overtime_impact": """
        SELECT OverTime,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
               ROUND(AVG(YearsAtCompany), 1) AS avg_tenure
        FROM employees
        GROUP BY OverTime
    """,
    "satisfaction_vs_attrition": """
        SELECT JobSatisfaction,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
        FROM employees
        GROUP BY JobSatisfaction
        ORDER BY JobSatisfaction
    """,
    "tenure_attrition": """
        SELECT TenureGroup,
               COUNT(*) AS total_employees,
               SUM(AttritionFlag) AS attrition_count,
               ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
               ROUND(AVG(MonthlyIncome), 0) AS avg_income,
               ROUND(AVG(SatisfactionScore), 2) AS avg_satisfaction
        FROM employees
        GROUP BY TenureGroup
        ORDER BY TenureGroup
    """,
    "high_risk_employees": """
        SELECT EmployeeNumber, Department, JobRole, Age,
               MonthlyIncome, YearsAtCompany, JobSatisfaction,
               EnvironmentSatisfaction, OverTime, Attrition,
               ROUND(SatisfactionScore, 2) AS satisfaction_score
        FROM employees
        WHERE JobSatisfaction <= 2
          AND EnvironmentSatisfaction <= 2
          AND OverTime = 'Yes'
        ORDER BY MonthlyIncome ASC
    """,
    "summary_stats": """
        SELECT
            COUNT(*) AS total_employees,
            SUM(AttritionFlag) AS total_attrition,
            ROUND(SUM(AttritionFlag)*100.0/COUNT(*), 1) AS attrition_rate_pct,
            ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
            ROUND(AVG(CASE WHEN AttritionFlag=1 THEN MonthlyIncome END), 0) AS avg_income_left,
            ROUND(AVG(CASE WHEN AttritionFlag=0 THEN MonthlyIncome END), 0) AS avg_income_stayed,
            ROUND(AVG(YearsAtCompany), 1) AS avg_tenure,
            ROUND(AVG(Age), 1) AS avg_age,
            SUM(OverTimeFlag) AS overtime_count,
            ROUND(AVG(SatisfactionScore), 2) AS avg_satisfaction
        FROM employees
    """
}

results = {}
for name, query in queries.items():
    results[name] = pd.read_sql_query(query, conn)
    results[name].to_csv(f"outputs/exports/{name}.csv", index=False)

conn.close()

stats     = results["summary_stats"].iloc[0]
dept      = results["attrition_by_department"]
role      = results["attrition_by_jobrole"]
tenure    = results["tenure_attrition"]
ot        = results["overtime_impact"]
high_risk = results["high_risk_employees"]

print(f"   Avg income (left)      : ${stats['avg_income_left']:,.0f}/mo")
print(f"   Avg income (stayed)    : ${stats['avg_income_stayed']:,.0f}/mo")
print(f"   Highest attrition dept : {dept.iloc[0]['Department']} ({dept.iloc[0]['attrition_rate_pct']}%)")
print(f"   Highest attrition role : {role.iloc[0]['JobRole']} ({role.iloc[0]['attrition_rate_pct']}%)")
print(f"   Overtime attrition     : {ot[ot['OverTime']=='Yes']['attrition_rate_pct'].values[0]}%")
print(f"   High risk employees    : {len(high_risk):,}")

# ── Step 3: Machine Learning ───────────────────────────────────────────────────
print("\n[3/6] Building attrition prediction models...")

df_ml = df.copy()
le = LabelEncoder()
cat_cols = ['Attrition','BusinessTravel','Department','EducationField',
            'Gender','JobRole','MaritalStatus','OverTime','TenureGroup','AgeGroup']
for col in cat_cols:
    if col in df_ml.columns:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Interaction features
df_ml['IncomeToAge']  = df_ml['MonthlyIncome'] / df_ml['Age']
df_ml['TenureRatio']  = df_ml['YearsAtCompany'] / (df_ml['TotalWorkingYears'] + 1)
df_ml['SatXOvertime'] = df['SatisfactionScore'] * df_ml['OverTimeFlag']
df_ml['PromotionLag'] = df_ml['YearsSinceLastPromotion'] / (df_ml['YearsAtCompany'] + 1)
df_ml['IncomeXSat']   = df_ml['MonthlyIncome'] * df['SatisfactionScore']
df_ml['AgeXTenure']   = df_ml['Age'] * df_ml['YearsAtCompany']
df_ml['OTXSat']       = df_ml['OverTimeFlag'] * df_ml['JobSatisfaction']
df_ml['DistXOT']      = df_ml['DistanceFromHome'] * df_ml['OverTimeFlag']

drop_cols    = ['AttritionFlag', 'IncomePerYear', 'SatisfactionScore']
feature_cols = [c for c in df_ml.columns if c not in drop_cols + ['Attrition']]

X = df_ml[feature_cols]
y = df_ml['AttritionFlag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Oversample minority class to 50/50
X_tr_df     = pd.concat([X_train, y_train], axis=1)
majority    = X_tr_df[X_tr_df.AttritionFlag == 0]
minority    = X_tr_df[X_tr_df.AttritionFlag == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced    = pd.concat([majority, minority_up])
X_tr_bal    = balanced.drop('AttritionFlag', axis=1)
y_tr_bal    = balanced['AttritionFlag']

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr_bal)
X_te_sc  = scaler.transform(X_test)

print("   Training 5 models (this may take ~60 seconds)...")

lr_model  = LogisticRegression(max_iter=2000, random_state=42, C=0.01)
rf_model  = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
gb_model  = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                        max_depth=4, subsample=0.8, random_state=42)
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, class_weight='balanced'),
    n_estimators=300, learning_rate=0.05, random_state=42)

lr_model.fit(X_tr_sc,  y_tr_bal)
rf_model.fit(X_tr_bal, y_tr_bal)
gb_model.fit(X_tr_bal, y_tr_bal)
ada_model.fit(X_tr_bal, y_tr_bal)

stacking = StackingClassifier(
    estimators=[
        ('lr',  Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=2000, C=0.01, random_state=42))])),
        ('rf',  RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb',  GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=4, subsample=0.8, random_state=42)),
        ('ada', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, class_weight='balanced'), n_estimators=200, learning_rate=0.05, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    cv=StratifiedKFold(5), n_jobs=-1
)
stacking.fit(X_tr_bal, y_tr_bal)

lr_p    = lr_model.predict_proba(X_te_sc)[:, 1]
rf_p    = rf_model.predict_proba(X_test)[:, 1]
gb_p    = gb_model.predict_proba(X_test)[:, 1]
ada_p   = ada_model.predict_proba(X_test)[:, 1]
stack_p = stacking.predict_proba(X_test)[:, 1]
ens_p   = (0.40*lr_p + 0.25*gb_p + 0.20*rf_p + 0.15*ada_p)

thr = 0.38
models_eval = {
    'Logistic Regression': (lr_p,    (lr_p    > thr).astype(int)),
    'Random Forest':        (rf_p,    (rf_p    > thr).astype(int)),
    'Gradient Boosting':    (gb_p,    (gb_p    > thr).astype(int)),
    'AdaBoost':             (ada_p,   (ada_p   > thr).astype(int)),
    'Stacking Classifier':  (stack_p, (stack_p > thr).astype(int)),
    'Weighted Ensemble':    (ens_p,   (ens_p   > thr).astype(int)),
}

model_results = []
print(f"\n   {'Model':<25} {'AUC-ROC':>8} {'AUPRC':>8} {'Recall':>8} {'F1':>6}")
print("   " + "-"*57)

for name, (probs, preds) in models_eval.items():
    auc    = roc_auc_score(y_test, probs)
    auprc  = average_precision_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    recall = report.get('1', {}).get('recall', 0)
    f1     = report.get('1', {}).get('f1-score', 0)
    print(f"   {name:<25} {auc:>8.4f} {auprc:>8.4f} {recall:>8.4f} {f1:>6.4f}")
    model_results.append({'Model': name, 'AUC_ROC': auc, 'AUPRC': auprc,
                          'Recall': recall, 'F1': f1})

model_df = pd.DataFrame(model_results)
model_df.to_csv("outputs/exports/model_comparison.csv", index=False)

feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
feat_imp.to_csv("outputs/exports/feature_importance.csv", index=False)

# Score all employees with weighted ensemble
lr_all  = lr_model.predict_proba(scaler.transform(X))[:, 1]
rf_all  = rf_model.predict_proba(X)[:, 1]
gb_all  = gb_model.predict_proba(X)[:, 1]
ada_all = ada_model.predict_proba(X)[:, 1]
ens_all = (0.40*lr_all + 0.25*gb_all + 0.20*rf_all + 0.15*ada_all)

df['AttritionRisk'] = ens_all.round(4)
df['RiskTier'] = pd.cut(df['AttritionRisk'],
                          bins=[0, 0.3, 0.5, 0.7, 1.0],
                          labels=['Low', 'Medium', 'High', 'Critical'])
df[['EmployeeNumber','Department','JobRole','Age','MonthlyIncome',
    'YearsAtCompany','JobSatisfaction','OverTime','Attrition',
    'AttritionRisk','RiskTier']].to_csv("outputs/exports/scored_employees.csv", index=False)

# ── Step 4: Visualizations ─────────────────────────────────────────────────────
print("\n[4/6] Generating visualizations...")

# Chart 1: Attrition Overview
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Employee Attrition Overview', fontsize=14, fontweight='bold')
axes[0].pie([retain, attrition],
             labels=[f'Stayed\n{retain}', f'Left\n{attrition}'],
             colors=[RETAIN_COLOR, ATTRITION_COLOR],
             autopct='%1.1f%%', startangle=90,
             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Overall Attrition Rate')
dept_data = results["attrition_by_department"]
axes[1].barh(dept_data['Department'], dept_data['attrition_rate_pct'],
              color=[ATTRITION_COLOR, WARN_COLOR, RETAIN_COLOR], edgecolor='white')
axes[1].set_xlabel('Attrition Rate (%)')
axes[1].set_title('Attrition Rate by Department')
for i, v in enumerate(dept_data['attrition_rate_pct']):
    axes[1].text(v + 0.3, i, f'{v}%', va='center', fontweight='bold')
age_data = results["attrition_by_age_group"]
axes[2].bar(age_data['AgeGroup'], age_data['attrition_rate_pct'],
             color=ATTRITION_COLOR, alpha=0.8, edgecolor='white')
axes[2].set_xlabel('Age Group')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Attrition Rate by Age Group')
plt.tight_layout()
plt.savefig("outputs/charts/1_attrition_overview.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Job Role Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Attrition by Job Role', fontsize=14, fontweight='bold')
role_data = results["attrition_by_jobrole"].sort_values('attrition_rate_pct')
colors_role = [ATTRITION_COLOR if v > 20 else WARN_COLOR if v > 10
               else RETAIN_COLOR for v in role_data['attrition_rate_pct']]
axes[0].barh(role_data['JobRole'], role_data['attrition_rate_pct'],
              color=colors_role, edgecolor='white')
axes[0].set_xlabel('Attrition Rate (%)')
axes[0].set_title('Attrition Rate by Role')
axes[0].axvline(x=attr_rate, color='gray', linestyle='--', alpha=0.7,
                label=f'Avg ({attr_rate:.1f}%)')
axes[0].legend(fontsize=8)
role_income = results["attrition_by_jobrole"].sort_values('avg_monthly_income')
axes[1].barh(role_income['JobRole'], role_income['avg_monthly_income'],
              color=ACCENT_COLOR, alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Avg Monthly Income ($)')
axes[1].set_title('Avg Monthly Income by Role')
plt.tight_layout()
plt.savefig("outputs/charts/2_jobrole_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Key Attrition Drivers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Key Attrition Drivers', fontsize=14, fontweight='bold')
ot_data  = results["overtime_impact"]
sat_data = results["satisfaction_vs_attrition"]
axes[0].bar(ot_data['OverTime'], ot_data['attrition_rate_pct'],
             color=[RETAIN_COLOR, ATTRITION_COLOR], width=0.4, edgecolor='white')
axes[0].set_xlabel('Overtime Required')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].set_title('Overtime vs Attrition Rate')
for i, (_, row) in enumerate(ot_data.iterrows()):
    axes[0].text(i, row['attrition_rate_pct'] + 0.5,
                 f"{row['attrition_rate_pct']}%", ha='center', fontweight='bold')
axes[1].bar(sat_data['JobSatisfaction'].astype(str), sat_data['attrition_rate_pct'],
             color=[ATTRITION_COLOR, WARN_COLOR, ACCENT_COLOR, RETAIN_COLOR], edgecolor='white')
axes[1].set_xlabel('Job Satisfaction (1=Low, 4=High)')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Job Satisfaction vs Attrition Rate')
for i, v in enumerate(sat_data['attrition_rate_pct']):
    axes[1].text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/3_attrition_drivers.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Income & Tenure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Income & Tenure Analysis', fontsize=14, fontweight='bold')
left_income   = df[df['AttritionFlag'] == 1]['MonthlyIncome']
stayed_income = df[df['AttritionFlag'] == 0]['MonthlyIncome']
axes[0].hist(stayed_income, bins=40, color=RETAIN_COLOR, alpha=0.7,
              label=f'Stayed (n={len(stayed_income):,})', density=True)
axes[0].hist(left_income, bins=40, color=ATTRITION_COLOR, alpha=0.7,
              label=f'Left (n={len(left_income):,})', density=True)
axes[0].axvline(stayed_income.mean(), color=RETAIN_COLOR, linestyle='--', linewidth=2,
                label=f'Stayed avg ${stayed_income.mean():,.0f}')
axes[0].axvline(left_income.mean(), color=ATTRITION_COLOR, linestyle='--', linewidth=2,
                label=f'Left avg ${left_income.mean():,.0f}')
axes[0].set_xlabel('Monthly Income ($)')
axes[0].set_ylabel('Density')
axes[0].set_title('Income Distribution by Attrition')
axes[0].legend(fontsize=8)
tenure_data = results["tenure_attrition"]
x = range(len(tenure_data))
axes[1].bar(x, tenure_data['attrition_rate_pct'], color=ATTRITION_COLOR, alpha=0.8, edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(tenure_data['TenureGroup'], rotation=15)
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Attrition Rate by Tenure Group')
for i, v in enumerate(tenure_data['attrition_rate_pct']):
    axes[1].text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig("outputs/charts/4_income_tenure_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Feature Importance
fig, ax = plt.subplots(figsize=(12, 6))
colors_feat = [ATTRITION_COLOR if i < 5 else ACCENT_COLOR for i in range(len(feat_imp))]
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors_feat, edgecolor='white')
ax.set_xlabel('Feature Importance Score')
ax.set_title('Top 15 Predictors of Employee Attrition (Random Forest)',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
red_p   = mpatches.Patch(color=ATTRITION_COLOR, label='Top 5 predictors')
green_p = mpatches.Patch(color=ACCENT_COLOR,    label='Supporting features')
ax.legend(handles=[red_p, green_p])
plt.tight_layout()
plt.savefig("outputs/charts/5_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Model Comparison -- all 6 models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Attrition Prediction Model Performance', fontsize=14, fontweight='bold')
bar_colors = ['#457B9D','#2A9D8F','#E9C46A','#E76F51','#6A4C93','#E63946']
for ax, metric, title in zip(axes, ['AUC_ROC','AUPRC','Recall'],
                                    ['AUC-ROC','AUPRC','Recall (Attrition Class)']):
    bars = ax.bar(model_df['Model'], model_df[metric],
                   color=bar_colors[:len(model_df)], alpha=0.9, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(model_df['Model'], rotation=25, ha='right', fontsize=7)
    for bar, val in zip(bars, model_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/6_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Step 5: Summary ────────────────────────────────────────────────────────────
print("\n[5/6] Exports saved to outputs/exports/")
print("\n[6/6] Done!")
print(f"\n   CHARTS  -> outputs/charts/")
print(f"   EXPORTS -> outputs/exports/")
best_auc    = model_df.loc[model_df['AUC_ROC'].idxmax()]
best_recall = model_df.loc[model_df['Recall'].idxmax()]
print(f"\n   Best AUC model      : {best_auc['Model']} ({best_auc['AUC_ROC']:.4f})")
print(f"   Best recall model   : {best_recall['Model']} ({best_recall['Recall']:.4f})")
print(f"   High risk flagged   : {len(df[df['RiskTier'].isin(['High','Critical'])]):,}")
print("\n" + "=" * 60)
print("  Connect outputs/exports/ to Power BI for live dashboard")
print("=" * 60)
