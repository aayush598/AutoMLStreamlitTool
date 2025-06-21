
# 🤖 AutoML CSV Trainer & Tester

A no-code **Streamlit-based Machine Learning tool** that allows anyone — even with minimal technical knowledge — to upload a CSV file, train/test a machine learning model, and visualize results interactively. It supports automated data preprocessing, model training, testing, evaluation, and visual reporting.

---

## 📂 Features

✅ Train models automatically on CSV data  
✅ Select models and see evaluation metrics  
✅ Test trained models with test CSVs  
✅ Auto-generated accuracy, confusion matrix, and graphs  
✅ No coding required — user-friendly web interface  
✅ Exportable trained `.pkl` model  
✅ Works as Web App and Standalone `.exe`  
✅ Docker support for deployment

---

## 📦 Folder Structure

```

automl-csv-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── config.toml
├── src/
│   ├── constants.py
│   ├── trainer.py
│   ├── tester.py
│   ├── utils.py
│   └── visualizer.py

````

---

## 🚀 Run Locally (Ubuntu / Mac / Windows)

```bash
# Clone the repository
git clone https://github.com/your-username/automl-csv-app.git
cd automl-csv-app

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
````

Then go to: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Example CSV Format

Make sure your input CSV has the target column (label) as the **last column**.

```
feature1, feature2, ..., label
value1,   value2,   ..., classA
```

---

## 🐳 Run with Docker (Ubuntu/Linux)

```bash
# Build Docker image
docker build -t automl-csv-app .

# Run Docker container
docker run -d -p 8501:8501 --name automl_tool automl-csv-app

# Access app at:
http://localhost:8501
```

---

## 💻 Convert to `.exe` (Windows)

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Create `.exe` using PyInstaller

```bash
pyinstaller app.py --onefile --noconsole --hidden-import=sklearn
```

### 3. (Optional) Edit `app.spec` to bundle `.streamlit/config.toml`:

```python
# In Analysis block:
datas=[('.streamlit/config.toml', '.streamlit')],
```

### 4. Rebuild `.exe`

```bash
pyinstaller app.spec
```

### ✅ Result

Your executable will be created in:

```
dist/
└── AutoMLTrainer.exe
```

You can now distribute this `.exe` to others for running locally without Python installed!



## 🧠 Future Enhancements

* Model comparison dashboard
* Feature selection and engineering options
* Deep learning support (Keras/PyTorch)
* Advanced hyperparameter tuning

---

## 👨‍💻 Developed By

**Aayush Gid**
B.Tech Electronics & Communication | Embedded Systems & AI/ML
Connect on [LinkedIn](https://www.linkedin.com/) *(Update link)*

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it for personal or commercial purposes.
