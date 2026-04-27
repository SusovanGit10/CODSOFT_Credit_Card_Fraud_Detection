💳 SmartBank Fraud Detection System










An end-to-end Machine Learning + Web App that detects fraudulent credit card transactions using behavioral patterns and transaction data.

🚀 Live Demo

👉 https://your-app-name.streamlit.app

🖼️ Screenshots
📊 Dashboard

🔍 Fraud Detection Interface

📈 Risk Output

📜 Transaction History

📌 Tip: Add screenshots to assets/ folder in your repo

✨ Features
🔍 Real-time fraud prediction
📊 Risk scoring (probability-based)
🧠 Feature engineering (Age from DOB, time patterns)
🌍 Location-aware inputs (State → City)
🔎 Searchable fields (Merchant, Job)
📜 Transaction history tracking
📊 Dashboard with insights
🧠 Machine Learning Pipeline
Models Trained
Logistic Regression
Decision Tree
Random Forest (Selected Best)
Techniques Used
SMOTE (handling class imbalance)
Feature Engineering (time, age, behavior)
Threshold tuning (precision vs recall trade-off)
Model comparison & evaluation
📊 Model Performance
Model	Precision	Recall	F1 Score	ROC-AUC
Logistic Regression	Low	High	Low	Good
Decision Tree	Medium	High	Medium	Very Good
Random Forest	Best Balance	High	Best	Excellent
📁 Project Structure
credit-card-fraud-detection/
│
├── app.py
├── main.py
├── requirements.txt
│
├── models/
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── features.pkl
│   ├── options.pkl
│   └── state_city_map.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
└── assets/
    ├── dashboard.png
    ├── detection.png
    ├── risk.png
    └── history.png
⚙️ Installation
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
▶️ Run the App
streamlit run app.py
🧪 Example Output
Fraud Probability Score
Risk Classification (Low / Medium / High)
AI-based Insights
🛠️ Tech Stack
Python
Scikit-learn
Pandas
Streamlit
Joblib
🎯 Use Cases
Banking fraud detection
Fintech risk analysis
Payment anomaly detection
Real-time transaction monitoring
📌 Future Improvements
🔐 Authentication system
📊 Advanced analytics dashboard
⚡ FastAPI backend integration
🌍 Multi-country support
👨‍💻 Author

Susovan Hati

📜 License

This project is licensed under the MIT License.