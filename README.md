# Titanic Survival Lab (Flask)

A classroom demo that trains a logistic regression classifier on the Kaggle Titanic training data, reports a held-out validation score, predicts a passenger's survival probability, and writes predictions for `test.csv` in Kaggle submission format.

## Run locally

```bash
cd titanic-flask
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

## Deploy

The included `render.yaml`, `Procfile`, and `requirements.txt` are ready for a Python WSGI host such as Render. Push this folder to a Git repository, connect that repository in Render, and deploy the Blueprint. The service listens on the host-provided port through Gunicorn; Render generates the production `SECRET_KEY`.

## Model notes

- Features: passenger class, sex, age, siblings/spouse, parents/children, fare, and embarkation port.
- Missing values: median imputation for numeric features and most-frequent imputation for categories.
- Validation: stratified 80/20 split with a fixed random seed. Metrics are calculated before refitting on all labeled rows.
- The app includes a held-out validation confusion matrix, precision, recall, F1, and accuracy.
- Uploaded training CSVs must contain `Survived` and the model feature columns. Uploads are read into memory and are not saved to disk.
- `gender_submission.csv` is not training data; it contains baseline predictions without ground-truth labels.
