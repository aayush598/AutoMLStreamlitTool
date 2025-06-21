# 🧠 AutoML CSV Trainer & Tester (Streamlit Tool)

An easy-to-use Streamlit-based AutoML tool that allows anyone to **train** and **test** machine learning models on CSV datasets — no coding required!

## 🚀 Features

- 📂 Upload CSV data for training or testing
- ⚙️ Automatic preprocessing (missing values, encoding, scaling)
- 🤖 Automatic ML model training using multiple algorithms
- 📊 Model evaluation with accuracy, precision, recall, F1-score
- 🧾 Confusion matrix & classification report visualizations
- 💾 Model saving and loading using `joblib`
- 🖼️ Streamlit-based interactive web interface
- 🧱 `.exe` builder for offline desktop usage (via PyInstaller)

---

## 📁 Project Structure

```bash
.
├── app.py                     # Main Streamlit app
├── requirements.txt
├── .streamlit/
│   └── config.toml            # UI config
├── outputs/                   # Saved models and plots
├── src/
│   ├── constants.py           # Global constants
│   ├── tester.py              # Model testing logic
│   ├── trainer.py             # Model training logic
│   ├── utils.py               # Preprocessing helpers
│   └── visualizer.py          # Plotting functions
├── setup/
│   └── build_spec.py          # PyInstaller packaging script
└── auto_py_to_exe.json        # Optional GUI-based .exe config
````

---

## 💻 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/automl-csv-trainer.git
cd automl-csv-trainer
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

Then visit: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Usage

1. Choose between **Train Model** or **Test Model**
2. Upload your `.csv` file
3. For training:

   * Select target column
   * Tool trains multiple classifiers and evaluates them
   * Best model is saved to `/outputs/`
4. For testing:

   * Upload a previously trained `.joblib` model
   * Upload new test CSV
   * Get evaluation report and visualizations

---

## 📦 Convert to .EXE (Optional)

To generate a standalone `.exe` for local usage:

### Option 1: Using Python Script

```bash
python setup/build_spec.py
```

### Option 2: Using auto-py-to-exe GUI

```bash
auto-py-to-exe
```

Then load `auto_py_to_exe.json` and convert.

---

## 📊 Sample Output

* 📈 Confusion matrix (heatmap)
* 📃 Classification report
* ✅ Accuracy, Precision, Recall, F1
* 📦 `.joblib` model file

---

## 📌 Dependencies

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Aayush Gid**
B.Tech, Electronics & Communication
Streamlit / AI Automation / Embedded Enthusiast

---

