import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load part of the train data
csv_files = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]
samples = []
porcentaje_datos = 0.1

print("Loading data...")
for file in csv_files:
    # Leer el archivo CSV
    df = pd.read_csv(file)
    
    # Tomar una muestra aleatoria del porcentaje definido
    sample = df.sample(frac=porcentaje_datos, random_state=1234)
    
    # Agregar el sample a la lista
    samples.append(sample)
    del df
    gc.collect()

train = pd.concat(samples, ignore_index=True)

# Load the test data
testeo = pd.read_csv("ctr_test.csv")

# Identificar las columnas numéricas y categóricas
print("Identifying columns...")
categorical_columns = train.select_dtypes(include=['object']).columns
numerical_columns = train.select_dtypes(include=['number']).columns

# Train a tree on the train data
print("Separando X e Y...")
X = train.drop('Label', axis=1)
y = train['Label']

print("Splitting data en validation and training...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)

# Crear el ColumnTransformer para aplicar OneHotEncoder a las columnas categóricas y SimpleImputer a las numéricas
numerical_columns = train.select_dtypes(include=['number']).columns.difference(['Label'])
categorical_columns = train.select_dtypes(include=['object']).columns

print
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Definir el modelo y la pipeline
pipeline = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=2345))

# Definir el espacio de hiperparámetros
param_dist = {
    'decisiontreeclassifier__max_depth': randint(1, 30),
    'decisiontreeclassifier__min_samples_split': randint(2, 100),
    'decisiontreeclassifier__min_samples_leaf': randint(1, 100)
}

# Implementar RandomizedSearchCV
print("Implementing RandomizedSearchCV...")
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                   n_iter=5, scoring='roc_auc', random_state=1234, cv=3, verbose=1, error_score='raise')

print("Fitting the model...")
# Ajustar el modelo con los mejores hiperparámetros
random_search.fit(X_train, y_train)

# Obtener los mejores parámetros y el mejor AUC
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best AUC score: {best_score:.2f}")

# Predecir en el conjunto de validación
y_test_probs = random_search.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_test_probs)

print(f"AUC on validation set: {roc_auc:.2f}")

# Predecir en el conjunto de testeo
y_preds = random_search.predict_proba(testeo.drop(columns=["id"]))[:, 1]

# Make the submission file
submission_df = pd.DataFrame({"id": testeo["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("rand_search_ohe.csv", sep=",", index=False)