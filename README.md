# 🚢 Titanic Survival Prediction App

This project is a Streamlit web app that predicts whether a passenger would survive the Titanic disaster, based on input features like class, age, fare, and more.

## 🔧 Tech Stack

* Python
* Jupyter Notebook (for model training)
* scikit-learn
* joblib
* Streamlit

## 📁 Project Structure

```
strmlt_app/
│
├── app.py               # Streamlit web app
├── model.pkl            # Trained ML model
└── titanic.csv          # Training dataset (optional)
```

## ✅ Features

* Takes in user inputs like Pclass, Age, Sex, etc.
* Predicts survival using a pre-trained model
* Runs directly in the browser using Streamlit

## 🚀 How to Run

1. Clone or download this repo

2. Make sure Python and pip are installed

3. Install Streamlit (if not already):

   ```bash
   pip install streamlit
   ```

4. Navigate to the project folder in terminal:

   ```bash
   cd path/to/strmlt_app
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

6. The app will open in your default browser at [http://localhost:8501/](http://localhost:8501/)

## ✍️ Author

Anthony – built and tested entirely in Jupyter Notebook.

---

Let me know if you'd like to add installation instructions for conda or a link to your dataset.
