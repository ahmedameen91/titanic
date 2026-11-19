from __future__ import annotations

import os
import pickle
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, send_file, url_for
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-classroom-demo-change-before-deploy")
model_state = {"model": None, "metrics": None, "source": "Kaggle train.csv"}


def make_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
    ])


def train_model(frame: pd.DataFrame) -> dict:
    missing = [column for column in FEATURES + ["Survived"] if column not in frame.columns]
    if missing:
        raise ValueError(
            "This file is missing required columns: " + ", ".join(missing)
            + ". Use the Kaggle train.csv, which includes passenger features and Survived labels."
        )
    data = frame[FEATURES + ["Survived"]].copy()
    data["Survived"] = pd.to_numeric(data["Survived"], errors="coerce")
    data = data[data["Survived"].isin([0, 1])]
    if len(data) < 20 or data["Survived"].nunique() != 2:
        raise ValueError("Training needs at least 20 labeled passengers and both Survived classes (0 and 1).")

    X, y = data[FEATURES], data["Survived"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    validation_model = make_pipeline()
    validation_model.fit(X_train, y_train)
    predictions = validation_model.predict(X_valid)
    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1])
    metrics = {
        "rows": int(len(data)),
        "survived": int(y.sum()),
        "validation_rows": int(len(y_valid)),
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "confusion": matrix.tolist(),
    }
    final_model = make_pipeline()
    final_model.fit(X, y)
    model_state.update(model=final_model, metrics=metrics)
    return metrics


def ensure_model() -> None:
    if model_state["model"] is None:
        train = pd.read_csv(DATA_DIR / "train.csv")
        train_model(train)


def clean_input(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Pclass": int(values.get("Pclass", 3)),
        "Sex": values.get("Sex", "male"),
        "Age": float(values.get("Age")) if values.get("Age") else None,
        "SibSp": int(values.get("SibSp", 0)),
        "Parch": int(values.get("Parch", 0)),
        "Fare": float(values.get("Fare")) if values.get("Fare") else None,
        "Embarked": values.get("Embarked", "S"),
    }], columns=FEATURES)


@app.get("/")
def home():
    ensure_model()
    return render_template(
        "index.html",
        metrics=model_state["metrics"],
        source=model_state["source"],
        prediction=None,
        form_values={},
    )


@app.post("/predict")
def predict():
    ensure_model()
    values = request.form.to_dict()
    try:
        passenger = clean_input(values)
        probability = float(model_state["model"].predict_proba(passenger)[0, 1])
        result = {"probability": probability, "survived": probability >= 0.5}
        return render_template("index.html", metrics=model_state["metrics"], source=model_state["source"], prediction=result, form_values=values)
    except (TypeError, ValueError) as error:
        flash(f"Please check the passenger values: {error}", "error")
        return render_template("index.html", metrics=model_state["metrics"], source=model_state["source"], prediction=None, form_values=values), 400


@app.post("/train")
def retrain():
    ensure_model()
    file = request.files.get("training_file")
    if not file or not file.filename:
        flash("Choose a training CSV first.", "error")
        return redirect(url_for("home"))
    try:
        frame = pd.read_csv(file)
        train_model(frame)
        model_state["source"] = file.filename
        flash(f"Model retrained from {file.filename} using a stratified 80/20 validation split.", "success")
    except Exception as error:
        flash(str(error), "error")
    return redirect(url_for("home"))


@app.get("/download-test-predictions.csv")
def download_predictions():
    ensure_model()
    test = pd.read_csv(DATA_DIR / "test.csv")
    missing = [column for column in FEATURES if column not in test.columns]
    if missing:
        return Response("test.csv is missing columns: " + ", ".join(missing), status=400)
    output = test[["PassengerId"]].copy()
    output["Survived"] = model_state["model"].predict(test[FEATURES]).astype(int)
    csv_bytes = output.to_csv(index=False).encode("utf-8")
    return Response(csv_bytes, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=titanic_predictions.csv"})


@app.get("/download-model.pkl")
def download_model():
    ensure_model()
    model_file = BytesIO(pickle.dumps(model_state["model"], protocol=pickle.HIGHEST_PROTOCOL))
    return send_file(
        model_file,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="titanic_survival_model.pkl",
    )


@app.get("/health")
def health():
    ensure_model()
    return {"status": "ok", "training_rows": model_state["metrics"]["rows"]}


if __name__ == "__main__":
    ensure_model()
    print("Titanic Survival Lab running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
