
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="ITB Mechanical Complication Predictor", page_icon="🧠")
bundle = joblib.load("itb_mechanical_model_v1.joblib")
model = bundle["model"]

st.title("ITB Mechanical Complication Predictor")
st.caption("Exploratory research prototype — not validated for clinical decision-making.")

st.warning(
    f"This model was developed from {bundle['training_n']} operations with only "
    f"{bundle['events']} mechanical complications ({bundle['event_rate']:.1%}). "
    "Individual probabilities are therefore uncertain and must not be used to determine patient care."
)

age = st.number_input("Age at surgery (years)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
previous_ops = st.number_input("Number of previous ITB operations", min_value=0, max_value=20, value=0, step=1)
previous_mech = st.selectbox("Previous mechanical ITB complication?", ["No", "Yes"])
indication = st.selectbox("Indication for current surgery", ["New insertion", "Replacement", "Revision", "Missing"])
placement = st.selectbox("Pump placement position", ["Subcutaneous", "Subfascial", "Submuscular", "Unknown", "Missing"])

if st.button("Estimate mechanical complication probability", type="primary"):
    x = np.array([[
        float(age),
        float(previous_ops),
        1 if previous_mech == "Yes" else 0,
        indication,
        placement
    ]], dtype=object)
    probability = float(model.predict_proba(x)[0,1])
    st.metric("Estimated probability", f"{probability*100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))
    st.info(
        "Interpret this as an exploratory model estimate, not a validated individual risk score. "
        "The model requires a larger number of mechanical events and external/prospective validation."
    )

with st.expander("Model performance"):
    st.write(f"Grouped 5-fold cross-validated ROC AUC: **{bundle['cv_auc']:.3f}**")
    st.write(f"Average precision: **{bundle['cv_average_precision']:.3f}**")
    st.write(f"Brier score: **{bundle['cv_brier']:.3f}**")
    st.write("Validation folds were grouped by patient so repeat operations from one patient did not cross between training and validation.")
