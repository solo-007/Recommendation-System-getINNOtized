# 🛒 E-commerce Recommendation System

![CI/CD](https://github.com/solo-007/Recommendation-System-getINNOtized/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/solo-007/Recommendation-System-getINNOtized)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.10%20%7C%203.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20App-brightgreen)

A full-stack **Recommendation System** project built with:
- 🐼 **Pandas / NumPy** for data processing  
- 🎯 **Scikit-learn** for ML models (CBF, CF, Hybrid, Clustering, Anomaly Detection)  
- 📊 **Matplotlib & Seaborn** for visualizations  
- 🌐 **Streamlit** for interactive deployment  

---

## 📂 Project Structure

```
Recommendation-System-getINNOtized/
│── app.py                 # Streamlit app (main entry)
│── recommendation.py      # Core recommendation functions
│── full_project.ipynb     # Full exploratory & modeling notebook
│── requirements.txt       # Dependencies
│── .github/workflows/ci.yml  # CI/CD pipeline config
│── data/                  # Data (loaded from Google Drive in app)
│── README.md              # Project documentation
```

---

## ⚙️ Features

- **Content-Based Filtering (CBF):** Recommends items based on item features (TF-IDF + cosine similarity).  
- **Collaborative Filtering (CF):** Learns from user-item interactions.  
- **Hybrid Model:** Combines CBF + CF for balanced results.  
- **User Segmentation:** K-Means clustering of user behavior.  
- **Anomaly Detection:** Isolation Forest to detect unusual/bot-like users.  
- **Category Graph:** Visualizes category hierarchy using `networkx`.  
- **Streamlit UI:** Interactive dashboard with multiple recommendation modes.  

---

## 🚀 Deployment

### 🔹 Run Locally
```bash
# Clone repository
git clone https://github.com/solo-007/Recommendation-System-getINNOtized.git
cd Recommendation-System-getINNOtized

# Create virtual environment & activate
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### 🔹 Deploy on Streamlit Cloud
1. Push to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → Deploy app.  
3. Use `app.py` as entry point.  

---

## 📊 Data Sources

Data is loaded from Google Drive (preprocessed CSVs):
- **Events dataset:** User interactions (views, add-to-cart, transactions).  
- **Item properties dataset:** Item features (metadata).  
- **Category tree dataset:** Hierarchical structure of categories.  

*(Original dataset is anonymized real-world e-commerce data.)*  

---

## ✅ CI/CD

This repo uses **GitHub Actions** for CI/CD:  
- ✅ Automatically installs dependencies.  
- ✅ Runs linting & tests.  
- ✅ Ensures app builds successfully before deployment.  

---

## 👨‍💻 Author & Supervision

- Developed by: **Solomon Sannie**  
- Supervised by: **Precious Darkwa**  
- Program: **AZUBI AFRICA**  

---

## 📜 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for more details.

