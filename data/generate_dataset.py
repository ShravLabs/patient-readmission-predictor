"""
generate_dataset.py
====================
Generates a synthetic patient readmission dataset that mirrors
the structure of the real patient_readmission.csv.

Run this if you want to regenerate or expand the dataset:
    python data/generate_dataset.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 25000

# ── Demographics ──────────────────────────────────────────────────────────────
age_brackets = ['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
ages = np.random.choice(age_brackets, N, p=[0.10, 0.18, 0.25, 0.27, 0.15, 0.05])

# ── Hospital stay ─────────────────────────────────────────────────────────────
time_in_hospital = np.random.randint(1, 15, N)
n_lab_procedures = np.random.randint(1, 100, N)
n_procedures     = np.random.randint(0, 10, N)
n_medications    = np.random.randint(1, 25, N)
n_outpatient     = np.random.randint(0, 20, N)
n_inpatient      = np.random.randint(0, 10, N)
n_emergency      = np.random.randint(0, 5, N)

# ── Clinical features ─────────────────────────────────────────────────────────
medical_specialties = ['InternalMedicine','Emergency/Trauma','Family/GeneralPractice',
                       'Cardiology','Surgery','Missing','Other']
medical_specialty = np.random.choice(medical_specialties, N,
                    p=[0.14, 0.08, 0.08, 0.06, 0.05, 0.50, 0.09])

diagnoses = ['Circulatory','Diabetes','Respiratory','Digestive','Injury','Other']
diag_1 = np.random.choice(diagnoses, N, p=[0.35,0.20,0.15,0.12,0.08,0.10])
diag_2 = np.random.choice(diagnoses, N, p=[0.25,0.18,0.18,0.14,0.10,0.15])
diag_3 = np.random.choice(diagnoses, N, p=[0.20,0.16,0.20,0.16,0.12,0.16])

glucose_test = np.random.choice(['no','normal','>200','>300'], N, p=[0.55,0.20,0.15,0.10])
A1Ctest      = np.random.choice(['no','normal','>7','>8'],     N, p=[0.50,0.20,0.20,0.10])
change       = np.random.choice(['no','yes'], N, p=[0.55,0.45])
diabetes_med = np.random.choice(['no','yes'], N, p=[0.25,0.75])

# ── Realistic readmission logic ───────────────────────────────────────────────
age_num_map = {'[40-50)':45,'[50-60)':55,'[60-70)':65,
               '[70-80)':75,'[80-90)':85,'[90-100)':95}
age_num = np.array([age_num_map[a] for a in ages])

readmit_prob = (
    0.30
    + 0.12 * (n_inpatient > 2).astype(float)
    + 0.10 * (n_emergency > 1).astype(float)
    + 0.08 * (time_in_hospital > 7).astype(float)
    + 0.07 * (age_num >= 70).astype(float)
    + 0.06 * (diag_1 == 'Diabetes').astype(float)
    + 0.05 * (A1Ctest == '>8').astype(float)
    + 0.04 * (glucose_test == '>300').astype(float)
    + 0.04 * (change == 'yes').astype(float)
    - 0.05 * (diabetes_med == 'yes').astype(float)
)
readmit_prob = np.clip(readmit_prob, 0, 1)
readmitted   = np.where(np.random.rand(N) < readmit_prob, 'yes', 'no')

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'age': ages, 'time_in_hospital': time_in_hospital,
    'n_lab_procedures': n_lab_procedures, 'n_procedures': n_procedures,
    'n_medications': n_medications, 'n_outpatient': n_outpatient,
    'n_inpatient': n_inpatient, 'n_emergency': n_emergency,
    'medical_specialty': medical_specialty,
    'diag_1': diag_1, 'diag_2': diag_2, 'diag_3': diag_3,
    'glucose_test': glucose_test, 'A1Ctest': A1Ctest,
    'change': change, 'diabetes_med': diabetes_med, 'readmitted': readmitted,
})

df.to_csv('data/patient_readmission.csv', index=False)
print(f"✅ Dataset generated: {df.shape}")
print(f"\nReadmission distribution:\n{df['readmitted'].value_counts()}")
print(f"Readmission rate: {(df['readmitted']=='yes').mean():.1%}")
