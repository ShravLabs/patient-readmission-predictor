"""
Patient Readmission Predictor
==============================
EDA + Feature Engineering + Model Training + SHAP Explainability
Uses: patient_readmission.csv (25,000 real hospital records)
Microsoft Data Science Internship Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

os.makedirs('outputs', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
BLUE   = '#0078D4'
ORANGE = '#E8750A'

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv('data/patient_readmission.csv')
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:\n{df['readmitted'].value_counts()}")
print(f"\nReadmission rate: {(df['readmitted']=='yes').mean():.1%}")

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────
df['target'] = (df['readmitted'] == 'yes').astype(int)

age_map = {
    '[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
    '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
    '[80-90)':85,'[90-100)':95
}
df['age_num'] = df['age'].map(age_map).fillna(65)

for col in ['change','diabetes_med']:
    df[col] = (df[col] == 'yes').astype(int)

test_map = {'no':0,'normal':1,'Norm':1,'>7':2,'>8':3}
df['glucose_test_enc'] = df['glucose_test'].map(test_map).fillna(0)
df['A1Ctest_enc']      = df['A1Ctest'].map(test_map).fillna(0)

le = LabelEncoder()
for col in ['medical_specialty','diag_1','diag_2','diag_3']:
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
df['visit_burden']    = df['n_inpatient'] + df['n_emergency']
df['med_per_day']     = df['n_medications'] / (df['time_in_hospital'] + 1)
df['is_elderly']      = (df['age_num'] >= 70).astype(int)
df['high_lab_work']   = (df['n_lab_procedures'] > 50).astype(int)
df['multi_diagnosis'] = ((df['diag_1'] != 'Other') & (df['diag_2'] != 'Other')).astype(int)

FEATURES = [
    'age_num','time_in_hospital','n_lab_procedures','n_procedures',
    'n_medications','n_outpatient','n_inpatient','n_emergency',
    'glucose_test_enc','A1Ctest_enc','change','diabetes_med',
    'medical_specialty_enc','diag_1_enc','diag_2_enc','diag_3_enc',
    'visit_burden','med_per_day','is_elderly','high_lab_work','multi_diagnosis'
]

X = df[FEATURES]
y = df['target']
print(f"\nFeatures: {len(FEATURES)} | Positive class: {y.mean():.1%}")

# ── 4. EDA VISUALS ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Patient Readmission – Exploratory Data Analysis', fontsize=16, fontweight='bold')

counts = df['readmitted'].value_counts()
axes[0,0].bar(['Not Readmitted','Readmitted'], counts.values, color=[BLUE,ORANGE], edgecolor='white')
axes[0,0].set_title('Target Distribution', fontweight='bold')
axes[0,0].set_ylabel('Patient Count')
for i,v in enumerate(counts.values):
    axes[0,0].text(i, v+100, f'{v:,}', ha='center', fontweight='bold')

age_order = ['[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
age_rate  = df[df['age'].isin(age_order)].groupby('age')['target'].mean()*100
age_rate  = age_rate.reindex(age_order)
axes[0,1].plot(age_rate.index, age_rate.values, marker='o', color=BLUE, linewidth=2.5, markersize=7)
axes[0,1].set_title('Readmission Rate by Age Group', fontweight='bold')
axes[0,1].set_ylabel('Readmission Rate (%)')
axes[0,1].tick_params(axis='x', rotation=30)

diag_rate = df.groupby('diag_1')['target'].mean().sort_values()*100
axes[0,2].barh(diag_rate.index, diag_rate.values, color=ORANGE, edgecolor='white')
axes[0,2].set_title('Readmission Rate by Primary Diagnosis', fontweight='bold')
axes[0,2].set_xlabel('Readmission Rate (%)')

for val,color,label in zip([0,1],[BLUE,ORANGE],['Not Readmitted','Readmitted']):
    axes[1,0].hist(df[df['target']==val]['time_in_hospital'], bins=14,
                   alpha=0.6, color=color, label=label, edgecolor='white')
axes[1,0].set_title('Hospital Stay Duration', fontweight='bold')
axes[1,0].set_xlabel('Days in Hospital')
axes[1,0].legend()

df.boxplot(column='n_medications', by='readmitted', ax=axes[1,1])
plt.sca(axes[1,1])
plt.title('Medications by Readmission Status', fontweight='bold')
axes[1,1].set_xlabel('Readmitted')
axes[1,1].set_ylabel('Number of Medications')

num_cols = ['age_num','time_in_hospital','n_lab_procedures','n_procedures',
            'n_medications','n_inpatient','n_emergency','visit_burden','target']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,2], cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':7})
axes[1,2].set_title('Feature Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ EDA saved → outputs/01_eda_overview.png")

# ── 5. TRAIN/TEST + SMOTE ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)
print(f"\nTrain (SMOTE): {X_train_res.shape} | Test: {X_test.shape}")

# ── 6. MODELS ─────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=150, random_state=42, eval_metric='logloss', verbosity=0),
}
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)
for name, model in models.items():
    X_tr = X_train_sc if name == 'Logistic Regression' else X_train_res
    cv_scores = cross_val_score(model, X_tr, y_train_res, cv=cv, scoring='roc_auc', n_jobs=-1)
    model.fit(X_tr, y_train_res)
    X_te   = X_test_sc if name == 'Logistic Regression' else X_test
    y_prob = model.predict_proba(X_te)[:,1]
    test_auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model':model,'y_prob':y_prob,'cv_auc':cv_scores.mean(),'test_auc':test_auc,'X_test':X_te}
    print(f"{name:25s} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test AUC: {test_auc:.4f}")

best_name = max(results, key=lambda k: results[k]['test_auc'])
best = results[best_name]
print(f"\n🏆 Best: {best_name} (AUC={best['test_auc']:.4f})")

# ── 7. MODEL COMPARISON PLOT ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['test_auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--',lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves'); axes[0].legend(loc='lower right', fontsize=9)
names = list(results.keys()); aucs = [results[n]['test_auc'] for n in names]
bars = axes[1].bar(names, aucs, color=[BLUE,ORANGE,'#107C10','#FFB900'], edgecolor='white')
axes[1].set_ylim(0.5,1.0); axes[1].set_ylabel('Test AUC'); axes[1].set_title('AUC Comparison')
axes[1].tick_params(axis='x', rotation=20)
for bar,auc in zip(bars,aucs):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{auc:.3f}', ha='center')
plt.tight_layout()
plt.savefig('outputs/02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Model comparison saved → outputs/02_model_comparison.png")

# ── 8. BEST MODEL EVAL ────────────────────────────────────────────────────────
y_pred = best['model'].predict(best['X_test']); y_prob = best['y_prob']
fig, axes = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle(f'{best_name} – Detailed Evaluation', fontsize=14, fontweight='bold')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Readmitted','Readmitted'], yticklabels=['Not Readmitted','Readmitted'])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
axes[1].plot(rec, prec, color=ORANGE, lw=2, label=f'AP={ap:.3f}')
axes[1].axhline(y_test.mean(), color='gray', linestyle='--', label='Baseline')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve'); axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/03_best_model_eval.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Best model eval saved → outputs/03_best_model_eval.png")
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, y_pred, target_names=['Not Readmitted','Readmitted']))

# ── 9. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
if hasattr(best['model'],'feature_importances_'):
    feat_imp = pd.Series(best['model'].feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9,6))
    feat_imp.plot(kind='barh', ax=ax, color=BLUE, edgecolor='white')
    ax.set_title(f'Top 15 Feature Importances – {best_name}', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('outputs/04_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance saved → outputs/04_feature_importance.png")

# ── 10. SHAP ──────────────────────────────────────────────────────────────────
print("\nGenerating SHAP values...")
try:
    explainer   = shap.TreeExplainer(best['model'])
    shap_values = explainer.shap_values(X_test.iloc[:300])
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    fig.suptitle('SHAP Explainability – Patient Readmission', fontsize=14, fontweight='bold')
    plt.sca(axes[0])
    shap.summary_plot(sv, X_test.iloc[:300], plot_type='bar', feature_names=FEATURES, show=False)
    axes[0].set_title('Mean |SHAP| Feature Importance', fontweight='bold')
    plt.sca(axes[1])
    shap.summary_plot(sv, X_test.iloc[:300], feature_names=FEATURES, show=False)
    axes[1].set_title('SHAP Beeswarm Plot', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/05_shap_explainability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP saved → outputs/05_shap_explainability.png")
except Exception as e:
    print(f"SHAP skipped: {e}")

print("\n" + "="*60)
print("✅ ALL DONE — check the outputs/ folder!")
print("="*60)
