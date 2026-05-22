import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ορισμός των βασικών classifiers που θα χρησιμοποιηθούν στο Stacking Ensemble

base_classifiers = [
    # 1. Οικογένεια: Δέντρα - Bagging
    (
        'rf', 
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ),
    # 2. Οικογένεια: Δέντρα - Boosting
    (
        'xgb', 
        XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            tree_method='hist',
            n_jobs=-1
        )
    ),
    # 3. Οικογένεια: Support Vector Machines
    (
        'svm', 
      # Το LinearSVC είναι πιο γρήγορο για μεγάλα datasets και είναι κατάλληλο για binary classification. Το dual=False είναι απαραίτητο όταν έχουμε περισσότερα δείγματα από χαρακτηριστικά.
        LinearSVC(dual=False, random_state=42, max_iter=2000)
    ),
    # 4. Οικογένεια: Probabilistic / Naive Bayes
    (
        'nb', 
        GaussianNB()
    )
]

# 1. ΦΟΡΤΩΣΗ & ΚΑΘΑΡΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
# Φορτώνουμε το dataset και κρατάμε μόνο τις στήλες που χρειαζόμαστε για την πρόβλεψη
df = pd.read_csv('1stproject.csv')
# Ορισμός της target μεταβλητής
target_col = 'disposition'

# Ορισμός των αριθμητικών και κατηγορικών στηλών που θα χρησιμοποιήσουμε
numeric_cols = ['age', 'triage_vital_hr', 'triage_vital_sbp', 'triage_vital_dbp']

# Οι κατηγορικές στήλες που θα κωδικοποιήσουμε με One-Hot Encoding
categorical_cols = [
    'gender', 'ethnicity', 'race', 'lang', 'religion', 'maritalstatus', 
    'employstatus', 'insurance_status', 'dep_name', 'arrivalmode', 
    'arrivalmonth', 'arrivalday', 'arrivalhour_bin', 'esi'
]

# Κρατάμε μόνο τις στήλες που χρειαζόμαστε για την πρόβλεψη
columns_to_keep = numeric_cols + categorical_cols + [target_col]
df = df[columns_to_keep]

df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. ΔΙΑΧΩΡΙΣΜΟΣ TRAIN-TEST SET (35% για test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

print("=== Πληροφορίες Διαχωρισμού Δεδομένων ===")
print(f"Συνολικό dataset: {df.shape[0]} γραμμές")
print(f"Training Set: {X_train.shape[0]} γραμμές ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test Set: {X_test.shape[0]} γραμμές ({len(X_test)/len(df)*100:.1f}%)")
print(f"Αριθμός Features: {X_train.shape[1]}\n")

# 3. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ (Preprocessing)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 4. ΕΚΠΑΙΔΕΥΣΗ STACKING ENSEMBLE (DIVERSE MODELS)
print("=== Base Classifiers για το Stacking ===")
for name, model in base_classifiers:
    print(f"Όνομα: {name} | Μοντέλο/Οικογένεια: {model.__class__.__name__}")

final_estimator = LogisticRegression(max_iter=2000, random_state=42)

# Το StackingClassifier θα συνδυάσει τις προβλέψεις των base classifiers και θα εκπαιδευτεί με τον Logistic Regression ως meta-learner
stacking_clf = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', stacking_clf)
])

# 5. ΕΚΠΑΙΔΕΥΣΗ ΤΟΥ ΠΛΗΡΟΥΣ PIPELINE
print("\nΞεκινάει η εκπαίδευση του Stacking Ensemble (XGBoost, RF, LinearSVC, NB)...")
full_pipeline.fit(X_train, y_train)
print("Η εκπαίδευση ολοκληρώθηκε επιτυχώς!\n")

# 6. ΠΡΟΒΛΕΨΗ ΚΑΙ ΑΞΙΟΛΟΓΗΣΗ ΣΤΟ TEST SET
y_pred = full_pipeline.predict(X_test)

# Αξιολόγηση με Classification Report και Confusion Matrix
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Οπτικοποίηση Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Stacking Classifier', fontsize=14)
plt.grid(False) 
plt.show()

# 7. ΥΠΟΛΟΓΙΣΜΟΣ ΚΑΙ ΟΠΤΙΚΟΠΟΙΗΣΗ LEARNING CURVE
print("\nΞεκινάει ο υπολογισμός του Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    estimator=full_pipeline,
    X=X_train,
    y=y_train,
    cv=3, 
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), 
    scoring='accuracy'
)
# Υπολογισμός μέσου όρου και τυπικής απόκλισης για τις καμπύλες εκπαίδευσης και επικύρωσης
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Οπτικοποίηση Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=6, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

# Η καμπύλη για το test set είναι πιο σημαντική για να δούμε αν υπάρχει overfitting ή underfitting
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=6, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve - Stacking Classifier (Diverse Models)', fontsize=16)
plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 8. ΑΞΙΟΛΟΓΗΣΗ ΣΕ ΝΕΟ DATASET (DOMAIN SHIFT)
# Εδώ θα φορτώσουμε το νέο dataset από το 3ο νοσοκομείο και θα κάνουμε πρόβλεψη χρησιμοποιώντας το ήδη εκπαιδευμένο μοντέλο.
print("\n" + "="*50)
print("Ξεκινάει η αξιολόγηση στο νέο dataset (3ο Νοσοκομείο)")
print("="*50)

# 1. Φόρτωση του νέου αρχείου
df_new = pd.read_csv('1stproject-TestSet.csv') 

# 2. Κρατάμε μόνο τις στήλες που χρειαζόμαστε
df_new = df_new[columns_to_keep]

# Αφαίρεση γραμμών που δεν έχουν disposition (αν υπάρχει η στήλη)
if target_col in df_new.columns:
    df_new = df_new.dropna(subset=[target_col])

# Διαχωρισμός Features και Target
if target_col in df_new.columns:
    X_new = df_new.drop(columns=[target_col])
    y_new = df_new[target_col]
else:
    X_new = df_new
    y_new = None

# Πρόβλεψη και Αξιολόγηση απευθείας μέσω του Pipeline
print("\nΓίνεται πρόβλεψη στα δεδομένα του 3ου νοσοκομείου...")
y_pred_new = full_pipeline.predict(X_new)

# Αν υπάρχει target variable, εμφανίζουμε την αξιολόγηση. Αν όχι, απλά αποθηκεύουμε τις προβλέψεις.
if y_new is not None:
    print("\n=== Classification Report (3ο Νοσοκομείο) ===")
    print(classification_report(y_new, y_pred_new))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_new, 
        y_pred_new, 
        cmap='Reds', 
        ax=ax
    )
    plt.title('Confusion Matrix - 3rd Hospital (Domain Shift)', fontsize=14)
    plt.grid(False)
    plt.show()
else:
    # Αν δεν υπάρχει target variable, απλά εμφανίζουμε τις προβλέψεις
    print("\nΤο νέο dataset δεν έχει target variable. Αποθήκευση προβλέψεων...")
    results = pd.DataFrame({'Predicted_Disposition': y_pred_new})
    results.to_csv('Predictions_3rd_Hospital.csv', index=False)
    print("Οι προβλέψεις αποθηκεύτηκαν στο 'Predictions_3rd_Hospital.csv'")