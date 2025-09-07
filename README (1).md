# 🛒 E-commerce Recommendation System

[![CI/CD](https://github.com/solo-007/Recommendation-System-getINNOtized/actions/workflows/ci.yml/badge.svg)](https://github.com/solo-007/Recommendation-System-getINNOtized/actions)

A production-ready **Recommendation System** built with **Streamlit, Scikit-learn, and Pandas**, deployed on **Streamlit Cloud**.  
This project explores multiple recommendation strategies, user segmentation, and anomaly detection to simulate a real-world e-commerce recommendation pipeline.  

---

## 📂 Project Structure

```
Recommendation-System-getINNOtized/
│
├── app.py                  # Main Streamlit application
├── recommendation.py       # Core recommendation logic
├── requirements.txt        # Dependencies (Streamlit Cloud + Local Dev)
├── README.md               # Project Documentation (this file)
│
├── .github/workflows/ci.yml  # CI/CD pipeline for testing + linting
└── data/                   # (Optional) Local dataset storage if not using Google Drive
```

---

## ⚙️ Features

✅ **Content-Based Filtering (CBF)** using TF-IDF & Nearest Neighbors  
✅ **Collaborative Filtering (CF)** optimized with Sparse Matrices  
✅ **Hybrid Recommendations** combining CF + CBF  
✅ **User Segmentation** with K-Means clustering  
✅ **Anomaly Detection** with Isolation Forest  
✅ **Interactive UI** powered by Streamlit  
✅ **Google Drive Dataset Integration** for seamless loading  

---

## 📊 Recommendation Modes

- **Content-Based Filtering** → Similar items given an `Item ID`  
- **Collaborative Filtering** → User-based recommendations given a `Visitor ID`  
- **Hybrid Model** → Combines CF + CBF for better accuracy  
- **User Segmentation** → Groups users into clusters based on behavior  
- **Anomaly Detection** → Detects abnormal or bot-like users  

---

## 📦 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/solo-007/Recommendation-System-getINNOtized.git
cd Recommendation-System-getINNOtized
```

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app locally:

```bash
streamlit run app.py
```

---

## 📂 Dataset Loading

Data is loaded **directly from Google Drive** in the app.  
If you prefer to download manually:

```bash
# Example: Download events dataset
gdown https://drive.google.com/uc?id=1pgf3rudMhoaOGB1MeIe41NIulxPq8okd -O data/events_df_filtered.csv

# Item properties
gdown https://drive.google.com/uc?id=1PveWfRDtgg5ytXdepZm2l_50n_quWbbB -O data/item_props_filtered.csv

# Category tree
gdown https://drive.google.com/uc?id=1dZ1eAvcPDY3RSuPBCezW9-WSRrT5uYSy -O data/category_tree.csv
```

The app samples **2% of the data** by default for performance, but you can adjust `sample_percentage` in `app.py`.

---

## 🚀 Deployment

This project is deployed on **Streamlit Cloud**.  
Every push to `main` triggers the **CI/CD pipeline** which:

1. Installs dependencies (`requirements.txt`)  
2. Runs basic tests (coming soon)  
3. Deploys automatically  

---

## 🧪 CI/CD

We use **GitHub Actions** for Continuous Integration & Deployment.  
The pipeline ensures your code runs correctly before deployment.  

Badge (see top of README):  
[![CI/CD](https://github.com/solo-007/Recommendation-System-getINNOtized/actions/workflows/ci.yml/badge.svg)](https://github.com/solo-007/Recommendation-System-getINNOtized/actions)

---

## 👨‍💻 Authors

- Developed by **Solomon Sannie**  
- Supervised by **Precious Darkwa**  
- **Azubi Africa Fellowship**

---

## 📜 License

This project is licensed under the MIT License – feel free to use it for learning and personal projects.
