# Concrete Compressive Strength Predictor

A Streamlit demonstration app using the trained HistGradientBoostingRegressor model from the Concrete Compressive Strength Prediction assignment.

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `concrete_strength_model.joblib`, and `requirements.txt` to the root of a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch (`main`), and set the main file path to `app.py`.
4. Deploy and wait for the app to finish building.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The scikit-learn version is pinned to 1.8.0 to match the environment in which the serialized model was generated. Do not change it without re-training and re-exporting the model.

## Important

This is an educational prototype, not an approved engineering design or quality-control tool. Confirm concrete strength through required laboratory tests and qualified engineering review.
