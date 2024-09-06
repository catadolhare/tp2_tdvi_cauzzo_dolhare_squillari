import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load part of the train data
train_data = pd.read_csv("ctr_20.csv")


# Load the test data
eval_data = pd.read_csv("ctr_test.csv")

# Train a tree on the train data
entrenamiento = train_data.sample(frac=0.8)
y_entrenamiento = entrenamiento["Label"]
x_entrenamiento = entrenamiento.drop(columns=["Label"])
x_entrenamiento = x_entrenamiento.select_dtypes(include='number')
gc.collect()

evaluacion = train_data.drop(entrenamiento.index)
y_evaluacion = evaluacion["Label"]
x_evaluacion = evaluacion.drop(columns=["Label"])
x_evaluacion = x_evaluacion.select_dtypes(include='number')
gc.collect()

best_auc = 0
best_depth = 0
best_split = 0
best_leaf = 0
best_pred = None

for i in range(1, 30):
    for j in range(100, 1000, 100):
        for l in range(100, 1000, 100):
            tree = DecisionTreeClassifier(max_depth=i, min_samples_split=j, min_samples_leaf=l, random_state=2345)
            tree.fit(x_entrenamiento, y_entrenamiento)
            
            y_test_preds = tree.predict(x_evaluacion)
            y_test_probs = tree.predict_proba(x_evaluacion)[:, tree.classes_ == 1].squeeze()
            roc_auc = tree.roc_auc_score(y_evaluacion, y_test_preds)

            if roc_auc > best_auc:
                best_score = roc_auc
                best_depth = i
                best_split = j
                best_leaf = l
                best_pred = y_test_probs

print(f'Best score: {best_auc:.2f}.')
print(f'Best depth: {best_depth}.')
print(f'Best split: {best_split}.')
print(f'Best leaf: {best_leaf}.')

best = DecisionTreeClassifier(max_depth=best_depth, min_samples_split=best_split, min_samples_leaf=best_leaf, random_state=2345)

# Predict on the evaluation set
eval_data = eval_data.select_dtypes(include='number')
y_preds = best.predict_proba(eval_data.drop(columns=["id"]))[:, best.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("hiperparametros.csv", sep=",", index=False)
