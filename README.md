# Recommendation System Project – getINNOtized
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
- Comprehensive project documentation
- Presentation file summarizing results & insights

---

## Project Structure

recommendation-system-getINNOtized/

├── data/                  # Raw and processed datasets

├── notebooks/             # Jupyter notebooks for EDA, preprocessing, training and modeling

├── models/                # Saved model files

├── visualizations/        # Charts, dashboards

├── presentation/          # Final slides (PPT and PDF)

├── README.md              # Project overview and documentation

├── .gitignore             # Python template

└── requirements.txt


---

## Getting Started
### Clone the repository
```bash
git clone https://github.com/solo-007/recommendation-system-getINNOtized.git
cd recommendation-system-getINNOtized
