# Property Recommender

This repository provides an end-to-end property recommendation system powered by machine learning and agentic AI. It includes training scripts, a FastAPI backend, and a Streamlit frontend.

## Prerequisites
- **Python 3.10** installed on your system.

## Setup

1. **Create and activate a virtual environment** (Python 3.10):

   ```bash
   python3.10 -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```
   
2. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Training the Model
Use the following command to train the scoring model:

  ```bash
  python train_model.py
  ```
This will process the data, train the model, and save the serialized pipeline to complex_price_model_v2.pkl

##Running the Streamlit App
Use the following command to launch the Streamlit frontend:

  ```bash
  streamlit run app_streamlit.py
  ```
The app will start at http://localhost:8501 by default.

## CV Results:
  MAE : 30413.43
  R2 : 0.949
  MAPE: 7.80%
