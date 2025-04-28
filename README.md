

markdown

# 🌍 Disaster Response Pipeline Project

This project uses machine learning to classify disaster response messages so they can be sent to the appropriate relief agency. It includes an ETL pipeline for data cleaning, an ML pipeline for classification, and a Flask web app for visualization and interaction.

---

## 📁 Project Structure

DISASTER_RESPONSE_PIPELINE_PROJECT/ │ ├── app/ │ ├── templates/ │ │ ├── go.html │ │ └── master.html │ └── run.py │ ├── data/ │ ├── disaster_categories.csv │ ├── disaster_messages.csv │ ├── DisasterResponse.db │ └── process_data.py │ ├── models/ │ ├── classifier.pkl │ └── train_classifier.py │ └── README.md

yaml
Copy
Edit

---

## 🛠️ How to Run

1. **ETL Pipeline** - Clean the data and save to SQLite database:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
ML Pipeline - Train a classifier and save the model:

bash
Copy
Edit
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the Web App:

bash
Copy
Edit
python app/run.py
Open your browser and navigate to:

cpp
Copy
Edit
http://0.0.0.0:3001/
📊 Features
Classifies disaster response messages into 36 categories.

Interactive web interface to test message classification.

Visualizes:

Distribution of message genres

Top 10 most frequent categories

⚙️ Tech Stack
Python, Pandas, NumPy

Scikit-learn (ML)

NLTK (text preprocessing)

SQLAlchemy (database)

Flask (web app)

Plotly (visualizations)

🔍 Model & Considerations
Uses a multi-output classification model (Random Forest).

Evaluated using precision, recall, F1-score.

Dataset is imbalanced—some categories (e.g., water) appear infrequently.

For emergency-critical labels, recall may be prioritized to avoid missing alerts.

💡 Ideas to Improve
Deploy the app to Heroku or Render.

Recommend NGOs to respond based on category predictions.

Add screenshots of the web interface and classification output.

Handle imbalance with resampling or class weighting.

Add word clouds, correlation heatmaps, or time series if timestamped data is available.

🙏 Acknowledgments
Dataset from Figure Eight (Appen).

Built as part of the Udacity Data Scientist Nanodegree.