import mlflow,mlflow.sklearn,joblib, os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import (
    load_data,
    drop_missing,
    fix_target,
    split_features_target,
    build_preprocessor
)



def get_next_model_version(base_path="model"):
    if not os.path.exists(base_path):
        return 1

    versions = [
        int(d.replace("v", ""))
        for d in os.listdir(base_path)
        if d.startswith("v") and d.replace("v", "").isdigit()
    ]

    if not versions:
        return 1

    return max(versions) + 1

def train():
    mlflow.set_experiment("heart-disease-prediction")

    with mlflow.start_run():
        # Load and preprocess data
        df = load_data("data/heart.csv")
        df = drop_missing(df)
        df = fix_target(df)

        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 🔑 MLflow logging
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="heart_disease_model"
        )

        version = get_next_model_version()
        model_dir = f"model/v{version}"
        os.makedirs(model_dir, exist_ok=True)

        model_path = f"{model_dir}/heart_model.pkl"
        joblib.dump(pipeline, model_path)

        print(f"Model Saved As Version : v{version}")

        print(f"Accuracy: {acc:.4f}")

        return pipeline, acc

def train_random_forest():
    mlflow.set_experiment("heart-disease-prediction")

    with mlflow.start_run():
        # Load and preprocess data
        df = load_data("data/heart.csv")
        df = drop_missing(df)
        df = fix_target(df)

        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # MLflow logging
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="heart_disease_model"
        )

        print(f"RandomForest Accuracy: {acc:.4f}")

        return pipeline, acc


if __name__ == "__main__":
    train()
    train_random_forest()
