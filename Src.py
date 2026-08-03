import pickle
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

with open(".\\Model\\model.pkl", "rb") as f:
    model = pickle.load(f)
with open(".\\Model\\OneHotEncoder.pkl", "rb") as f:
    ohe = pickle.load(f)

CATEGORICAL_COLS = list(ohe.feature_names_in_)


NUMERIC_COLS = [
    "age", "resting_bp_systolic", "resting_bp_diastolic", "cholesterol_total",
    "hdl", "ldl", "triglycerides", "fasting_blood_sugar", "hba1c", "bmi",
    "resting_heart_rate", "max_heart_rate_achieved", "exercise_induced_angina",
    "st_depression", "family_history", "alcohol_units_per_week",
    "exercise_minutes_per_week", "sleep_hours", "stress_score",
    "wearable_owner", "daily_steps", "diet_quality_score",
]

BOOL_COLS = {"exercise_induced_angina", "family_history", "wearable_owner"}

CATEGORY_CHOICES = dict(zip(CATEGORICAL_COLS, [list(c) for c in ohe.categories_]))
BOOL_CHOICES = ["False", "True"]

FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "resting_bp_systolic": "Resting BP (systolic)",
    "resting_bp_diastolic": "Resting BP (diastolic)",
    "cholesterol_total": "Total Cholesterol",
    "hdl": "HDL",
    "ldl": "LDL",
    "triglycerides": "Triglycerides",
    "fasting_blood_sugar": "Fasting Blood Sugar",
    "hba1c": "HbA1c",
    "bmi": "BMI",
    "resting_heart_rate": "Resting Heart Rate",
    "max_heart_rate_achieved": "Max Heart Rate Achieved",
    "chest_pain_type": "Chest Pain Type",
    "exercise_induced_angina": "Exercise Induced Angina",
    "st_depression": "ST Depression",
    "family_history": "Family History",
    "smoker_status": "Smoker Status",
    "alcohol_units_per_week": "Alcohol (units/week)",
    "exercise_minutes_per_week": "Exercise (min/week)",
    "sleep_hours": "Sleep Hours",
    "stress_score": "Stress Score",
    "wearable_owner": "Wearable Owner",
    "daily_steps": "Daily Steps",
    "diet_quality_score": "Diet Quality Score",
}


ALL_FIELDS = [
    "age", "sex", "resting_bp_systolic", "resting_bp_diastolic", "cholesterol_total",
    "hdl", "ldl", "triglycerides", "fasting_blood_sugar", "hba1c", "bmi",
    "resting_heart_rate", "max_heart_rate_achieved", "chest_pain_type",
    "exercise_induced_angina", "st_depression", "family_history", "smoker_status",
    "alcohol_units_per_week", "exercise_minutes_per_week", "sleep_hours",
    "stress_score", "wearable_owner", "daily_steps", "diet_quality_score",
]

widgets = {} 


def build_input_row(parent, row, col_name):
    """Create a label + input widget (Entry or Combobox) for one column."""
    label = ttk.Label(parent, text=FIELD_LABELS.get(col_name, col_name))
    label.grid(row=row, column=0, sticky="w", padx=8, pady=4)

    if col_name in CATEGORICAL_COLS:
        var = tk.StringVar(value=CATEGORY_CHOICES[col_name][0])
        widget = ttk.Combobox(
            parent, textvariable=var, values=CATEGORY_CHOICES[col_name],
            state="readonly", width=22,
        )
    elif col_name in BOOL_COLS:
        var = tk.StringVar(value=BOOL_CHOICES[0])
        widget = ttk.Combobox(
            parent, textvariable=var, values=BOOL_CHOICES,
            state="readonly", width=22,
        )
    else:
        var = tk.StringVar()
        widget = ttk.Entry(parent, textvariable=var, width=25)

    widget.grid(row=row, column=1, padx=8, pady=4)
    widgets[col_name] = var


def get_input_dataframe():
    """Read every field from the GUI and return a one-row DataFrame."""
    data = {}
    for col in ALL_FIELDS:
        raw_value = widgets[col].get().strip()
        if not raw_value:
            raise ValueError(f"Please fill in the '{FIELD_LABELS.get(col, col)}' field.")

        if col in BOOL_COLS:
            data[col] = raw_value == "True"
        elif col in CATEGORICAL_COLS:
            data[col] = raw_value
        else:
            try:
                data[col] = float(raw_value)
            except ValueError:
                raise ValueError(
                    f"'{FIELD_LABELS.get(col, col)}' must be a number, got: {raw_value}"
                )

    return pd.DataFrame([data])


def predict():
    try:
        df = get_input_dataframe()

        encoded = ohe.transform(df[CATEGORICAL_COLS])
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        encoded = np.asarray(encoded)    
        if encoded.ndim ==1:
            encoded =encoded.reshape(1,-1)        
                
        numeric = df[NUMERIC_COLS].astype(np.float32).values
        if numeric.ndim == 1:
            numeric = numeric.reshape(1,-1) 

        X = np.concatenate([numeric, encoded], axis=1)
        X.reshape(-1,1)
        prediction = model.predict(X)
        probability = float(np.ravel(prediction)[0])
        label = "High risk of heart disease" if probability >= 0.5 else "Low risk of heart disease"

        result_var.set(f"{label}  (probability: {probability:.2%})")
    except Exception as exc:
        messagebox.showerror("Error", str(exc))



window = tk.Tk()
window.title("Heart_Atack_Predictor")
window.geometry("520x600")
window.resizable(False, False)

container = ttk.Frame(window)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, borderwidth=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scroll_frame = ttk.Frame(canvas)

scroll_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

ttk.Label(scroll_frame, text="Heart Attack Predictor", font=("Segoe UI", 14, "bold")).grid(
    row=0, column=0, columnspan=2, pady=(10, 15)
)

for i, col in enumerate(ALL_FIELDS, start=1):
    build_input_row(scroll_frame, i, col)

predict_btn = ttk.Button(scroll_frame, text="Predict", command=predict)
predict_btn.grid(row=len(ALL_FIELDS) + 1, column=0, columnspan=2, pady=15)

result_var = tk.StringVar(value="")
result_label = ttk.Label(
    scroll_frame, textvariable=result_var, font=("Segoe UI", 11, "bold"), foreground="darkred"
)
result_label.grid(row=len(ALL_FIELDS) + 2, column=0, columnspan=2, pady=(0, 20))

window.mainloop()