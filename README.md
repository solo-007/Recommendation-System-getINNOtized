# Recommendation System Project – getINNOtized

![License](https://img.shields.io/github/license/solo-007/Recommendation-System-getINNOtized)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.10%20%7C%203.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20App-brightgreen)

<img width="768" height="577" alt="CRISP-DM" src="https://media.licdn.com/dms/image/C4E12AQGZXG-omsKv3g/article-cover_image-shrink_720_1280/0/1597499469493?e=2147483647&v=beta&t=QMg5FypP7FDW4wKoSaIEQnI34DeVM1NMO-uMyZ24kt0" />


## Overview
This project analyzes e-commerce event data to understand user behavior, predict item properties, identify abnormal users, and develop recommendation algorithms. The goal is to leverage these insights to enhance the product page experience and improve user engagement and conversion.
It follows the **CRISP-DM** methodology and explores recommendation approaches.

The project aims to:
• Find abnormal users of e-shop. 
• Generate features 
• Build a model 
• Create a metric that helps to evaluate quality of the model 

---

## Data Description
The analysis utilizes three datasets:

`events.csv`: Contains user interaction events with timestamps, visitor IDs, event types (view, addtocart, transaction), item IDs, and transaction IDs (for transaction events).

`item_properties.csv`: Contains item properties with timestamps, item IDs, property names, and property values.

`category_tree.сsv`: which describes category tree

---
## Tools & Technologies
| Tool/Library         | Purpose |
|----------------------|---------|
| Python               | Core programming language |
| Pandas, NumPy        | Data manipulation & preprocessing |
| Matplotlib, Seaborn  | Data visualization |
| scikit-learn         | Machine learning & evaluation |
| TF-IDF / Linear Kernel   | Building recommendation models |
| Git & GitHub         | Version control & project tracking |
| Streamlit         | for interactive deployment |

---

## 📋 CRISP-DM Methodology
1. **Business Understanding** – Define objectives & business questions
2. **Data Understanding** – Explore and profile the dataset
3. **Data Preparation** – Clean & prepare data for modeling
4. **Modeling** – Build Predictive Modeling and Recommendation Algorithms
6. **Evaluation** – Assess performance with metrics like Accuracy, Precision@K, Recall@K
7. **Deployment** – Present results & optional interactive demo

---

## Analytical Questions
1.	Which products/content are most frequently interacted with by users?
2.	Which user segments have similar purchase/viewing behaviors?
3.	What products/content have high engagement but low recommendation exposure?
4.	How do seasonal trends affect product/content interactions?
5.	What time of the day does purchase activity increases?
6.	What are the most frequent item id predicted?
7.	What is the percentage of visitors who made purchase?
8.	What are the total transactions over time?

---

## Deliverables
- Jupyter Notebooks with analysis & modeling
- Predictions and Abnormality detections
- Data visualizations answering business questions
- Streamlit app for Recommendation System
- Comprehensive project documentation
- Presentation file summarizing results & insights

---

## Project Structure

recommendation-system-getINNOtized/

│── .github/workflows/ci-cd.yml  # CI/CD pipeline config

├── data/                  # Raw and processed datasets

├── notebooks/             # Jupyter notebooks for EDA, preprocessing, training and modeling

├── models/                # Saved model files

├── visualizations/        # Charts, dashboards

├── presentation/          # Final slides (PPT and PDF)

│── app.py                 # Streamlit app (main entry)

├── README.md              # Project overview and documentation

├── .gitignore             # Python template

└── requirements.txt


---

## Features

- **Content-Based Filtering (CBF):** Recommends items based on item features (TF-IDF + cosine similarity).  
- **Collaborative Filtering (CF):** Learns from user-item interactions.  
- **Hybrid Model:** Combines CBF + CF for balanced results.  
- **User Segmentation:** K-Means clustering of user behavior.  
- **Anomaly Detection:** Isolation Forest to detect unusual/bot-like users.  
- **Category Graph:** Visualizes category hierarchy using `networkx`.  
- **Streamlit UI:** Interactive dashboard with multiple recommendation modes.  

---

## Getting Started

### Deployment

### Run Locally
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

### Deploy on Streamlit Cloud
1. Push to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → Deploy app.  
3. Use `app.py` as entry point.  

---

## CI/CD

This repo uses **GitHub Actions** for CI/CD:  
- Automatically installs dependencies.  
- Runs linting & tests.  
- Ensures app builds successfully before deployment.  

---

## Author & Supervision

- Developed by: **Solomon Sannie**  
- Supervised by: **Precious Darkwa**  
- Program: **AZUBI AFRICA**  

---

## License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for more details.
