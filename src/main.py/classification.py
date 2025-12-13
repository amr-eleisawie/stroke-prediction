import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


# ----------------- قراءة البيانات -----------------
data = pd.read_csv("C:\\Users\\user\\Downloads\\healthcare-dataset-stroke-data.csv")

# ----------------- معالجة القيم المفقودة -----------------
imputer = SimpleImputer(strategy='median')
data['bmi'] = imputer.fit_transform(data[['bmi']])

# ----------------- إزالة Outliers -----------------
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

data = remove_outliers(data, 'avg_glucose_level')
data = remove_outliers(data, 'bmi')

# ----------------- Log Transformation -----------------
data['avg_glucose_level'] = np.log1p(data['avg_glucose_level'])

# ----------------- معالجة المتغيرات الفئوية -----------------
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data['work_type'] = data['work_type'].replace({'Never_worked': 'Other'})
data['gender'] = data['gender'].replace({'Other': 'Male'})
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# ----------------- تقسيم البيانات -----------------
X = data.drop('stroke', axis=1)
y = data['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------- تعريف النماذج -----------------
models = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(binarize=1),
    "LogisticRegression": LogisticRegression(penalty='l2', solver='saga', max_iter=5000, random_state=33,class_weight='balanced'),
    "SGDClassifier": SGDClassifier(penalty='l2', loss='squared_error', learning_rate='optimal', random_state=33),
    "RandomForest": RandomForestClassifier(criterion='gini', n_estimators=300, max_depth=7, random_state=33),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=33),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVC": SVC(kernel='rbf', C=1.0, gamma='auto', max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=33),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform')
}
results =[]
# ----------------- تدريب وتقييم النماذج -----------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC-AUC handling
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = np.nan

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

# تحويل النتائج إلى DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1-score', ascending=False)
print(results_df)

# ----------------- Visualization -----------------
plt.figure(figsize=(12,6))
plt.bar(results_df['Model'], results_df['F1-score'], color='skyblue')
plt.xticks(rotation=45)
plt.ylabel('F1-score')
plt.title('Comparison of Models by F1-score')
plt.show()


