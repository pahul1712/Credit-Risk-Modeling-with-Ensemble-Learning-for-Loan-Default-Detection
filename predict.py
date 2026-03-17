import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load("models/xgb_credit_risk_model.pkl")
ohe = joblib.load("models/onehot_encoder.pkl")
scaler = joblib.load("models/standard_scaler.pkl")
onehot_columns = joblib.load("models/onehot_columns.pkl")
scaler_columns = joblib.load("models/scaler_columns.pkl")
final_feature_order = joblib.load("models/final_feature_order.pkl")


def prepare_input(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    # Feature engineering
    df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"]
    df["loan_to_emp_length_ratio"] = df["person_emp_length"] / df["loan_amnt"]
    df["int_rate_to_loan_amt_ratio"] = df["loan_int_rate"] / df["loan_amnt"]

    # One-hot encode categorical columns
    encoded_array = ohe.transform(df[onehot_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(onehot_columns))

    # Drop original categorical columns
    df_non_cat = df.drop(columns=onehot_columns).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    final_df = pd.concat([encoded_df, df_non_cat], axis=1)

    # Scale numerical columns
    final_df[scaler_columns] = scaler.transform(final_df[scaler_columns])

    # Match final training column order
    final_df = final_df.reindex(columns=final_feature_order, fill_value=0)

    return final_df


def predict_credit_risk(raw_input: dict):
    prepared_input = prepare_input(raw_input)

    pred = model.predict(prepared_input)[0]
    prob = model.predict_proba(prepared_input)[0][1]

    return {
        "predicted_class": int(pred),
        "default_probability": float(prob),
        "interpretation": "High-risk borrower" if pred == 1 else "Low-risk borrower"
    }


if __name__ == "__main__":
    sample_borrower = {
        "person_age": 28,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "C",
        "loan_amnt": 12000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }

    result = predict_credit_risk(sample_borrower)
    print(result)