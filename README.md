# ❤️ Heart Attack Risk Predictor

A machine learning application that predicts an individual's risk of heart disease based on clinical, lifestyle, and wearable-device data. The project includes a full training pipeline (Jupyter Notebook), a trained neural network model, a desktop GUI application, and a **web app** for real-time predictions.

**🔗 https://heart-attack-risk-predictor-mahan-liaghatmand.streamlit.app/

---

## 📋 Overview

This project uses a deep neural network trained on a dataset of ~9,000 patient records to classify heart disease risk as **high** or **low**, along with an estimated probability score. It combines both **medical/clinical indicators** (blood pressure, cholesterol, HbA1c, etc.) and **lifestyle/wearable data** (daily steps, sleep hours, stress score) to produce a well-rounded risk assessment.

A simple, easy-to-use **Tkinter desktop GUI** is included so that predictions can be made interactively without writing any code.

---

## ✨ Features

- **Neural network classifier** built with TensorFlow/Keras (4-layer dense architecture)
- **~90% test accuracy** with strong precision/recall balance
- **24 input features** spanning clinical measurements, lab results, and lifestyle habits
- **One-Hot Encoding** for categorical variables (sex, chest pain type, smoker status)
- **Interactive GUI** (Tkinter) for entering patient data and getting instant predictions
- **Web app** (Streamlit) — the same model, accessible from any browser via a shareable link
- **Pre-trained, ready-to-use model** — no retraining required to make predictions

---

## 🗂️ Project Structure

```
├── Dataset/Data.csv                  # Training dataset (~9,000 records)
├── Model/Heart_Atack_Predictor.ipynb # Model training & evaluation notebook
├── Model/model.pkl                   # Trained Keras neural network (pickled)
├── Model/OneHotEncoder.pkl           # Fitted OneHotEncoder for categorical features
├── Src.py                            # Tkinter desktop GUI application
├── app.py                            # Streamlit web app (same model, browser-based)
├── requirements.txt                  # Dependencies for the web app
└── .streamlit/config.toml            # Web app theme
```

---

## 🧠 Model Details

| Aspect | Description |
|---|---|
| **Architecture** | Sequential Dense Neural Network (31 → 16 → 8 → 1) |
| **Activation** | ReLU (hidden layers), Sigmoid (output layer) |
| **Optimizer** | Adam |
| **Loss Function** | Binary Crossentropy |
| **Regularization** | Early stopping (patience = 50, monitored on validation loss) |
| **Train/Test Split** | 80% / 20% |

### Performance

| Metric | Score |
|---|---|
| Test Accuracy | **90.5%** |
| Train Accuracy | 89.5% |
| Precision | 87.4% |
| Recall | 79.9% |

*(Metrics computed on the held-out 20% test split — see the notebook for the full confusion matrix.)*

---

## 📊 Input Features

The model uses **24 features**, split between numeric and categorical types:

**Numeric / Continuous:**
`age`, `resting_bp_systolic`, `resting_bp_diastolic`, `cholesterol_total`, `hdl`, `ldl`, `triglycerides`, `fasting_blood_sugar`, `hba1c`, `bmi`, `resting_heart_rate`, `max_heart_rate_achieved`, `st_depression`, `alcohol_units_per_week`, `exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `daily_steps`, `diet_quality_score`

**Boolean:**
`exercise_induced_angina`, `family_history`, `wearable_owner`

**Categorical (One-Hot Encoded):**
`sex`, `chest_pain_type`, `smoker_status`

**Target Variable:**
`has_heart_disease` (0 = No, 1 = Yes)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow
```

*(Tkinter ships with standard Python installations on most systems.)*

### Running the GUI Application

1. Make sure `model.pkl` and `OneHotEncoder.pkl` are placed inside a `Model/` subfolder (as referenced in `Src.py`), or update the file paths in `Src.py` to match your directory structure.
2. Run the application:

```bash
python Src.py
```

3. Fill in the patient's information in the form.
4. Click **Predict** to view the risk classification and probability score.

### Running the Web App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Retraining the Model

Open `Heart_Atack_Predictor.ipynb` in Jupyter Notebook or Google Colab to explore the data preprocessing, model architecture, training process, and evaluation metrics — or to retrain the model on new data.

---

## ☁️ Deploying the Web App (free, ~5 minutes)

**Option A — Streamlit Community Cloud (recommended, simplest):**
1. Push this repository to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically and gives you a public URL.

**Option B — Hugging Face Spaces:**
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose the **Streamlit** SDK, then either upload this project's files or connect the GitHub repo.
3. The Space builds automatically and gives you a public URL in the form `huggingface.co/spaces/<username>/<space-name>`.

Both options are free and don't require a credit card.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras** — model architecture & training
- **scikit-learn** — preprocessing (OneHotEncoder), train/test split, evaluation metrics
- **pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization (accuracy curves, confusion matrix)
- **Tkinter** — desktop GUI
- **Streamlit** — web app / browser-based GUI

---

## ⚠️ Disclaimer

This tool is built for **educational and demonstrative purposes only**. It is **not a certified medical diagnostic tool** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.

---

## 📄 License

This project is open for educational use. Feel free to fork, modify, and build upon it.
