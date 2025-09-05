import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# -------------------------------
# Google Drive File Links
# -------------------------------
# Replace FILE_ID with your actual file IDs
events_url = "https://drive.google.com/uc?id=1pgf3rudMhoaOGB1MeIe41NIulxPq8okd"
item_props_url = "https://drive.google.com/uc?id=1PveWfRDtgg5ytXdepZm2l_50n_quWbbB"
category_tree_url = "https://drive.google.com/uc?id=1dZ1eAvcPDY3RSuPBCezW9-WSRrT5uYSy"

@st.cache_data
def load_data(sample_percentage=0.02):
    try:
        events_df_filtered = pd.read_csv(events_url)
        item_props_filtered = pd.read_csv(item_props_url)
        category_tree = pd.read_csv(category_tree_url)

        # Sample for performance
        events_sample = events_df_filtered.sample(frac=sample_percentage, random_state=42)
        item_props_sample = item_props_filtered.sample(frac=sample_percentage, random_state=42)

        return events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None, None

# -------------------------------
# Load Data Automatically
# -------------------------------
st.info("📂 Loading data from Google Drive...")
events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree = load_data()

if events_df_filtered is None:
    st.stop()
else:
    st.success("✅ Data loaded successfully from Google Drive!")

    with st.expander("🔎 Data Preview"):
        st.subheader("Events Sample")
        st.dataframe(events_sample.head())
        st.subheader("Item Properties Sample")
        st.dataframe(item_props_sample.head())
        st.subheader("Category Tree")
        st.dataframe(category_tree.head())

# -------------------------------
# Content-Based Filtering
# -------------------------------
st.header("🎯 Content-Based Filtering (CBF)")

item_features = item_props_sample.groupby("itemid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(item_features["value"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(item_features.index, index=item_features["itemid"])

def recommend_content(itemid, top_n=5):
    if itemid not in indices:
        return []
    idx = indices[itemid]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    item_indices = [i[0] for i in sim_scores]
    return item_features.iloc[item_indices]["itemid"].tolist()

itemid_input = st.number_input("Enter an Item ID:", min_value=0, step=1, key="cbf")
if st.button("Get CBF Recommendations"):
    recs = recommend_content(itemid=itemid_input, top_n=5)
    st.write("📌 Recommended Items:", recs)

# -------------------------------
# Collaborative Filtering
# -------------------------------
st.header("👥 Collaborative Filtering (CF)")

weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
events_sample["weight"] = events_sample["event"].map(weight_map).fillna(0)

user_item_matrix = events_sample.pivot_table(
    index="visitorid", columns="itemid", values="weight", fill_value=0
)
user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_cf(visitorid, top_n=5):
    if visitorid not in user_item_matrix.index:
        return []
    similar_users = user_sim_df[visitorid].sort_values(ascending=False).index[1:]
    recommendations = events_sample[events_sample["visitorid"].isin(similar_users)] \
                        .groupby("itemid")["weight"].sum().sort_values(ascending=False)
    seen_items = set(events_sample[events_sample["visitorid"] == visitorid]["itemid"])
    recommendations = recommendations[~recommendations.index.isin(seen_items)]
    return recommendations.head(top_n).index.tolist()

visitorid_input = st.number_input("Enter a Visitor ID:", min_value=0, step=1, key="cf")
if st.button("Get CF Recommendations"):
    recs = recommend_cf(visitorid=visitorid_input, top_n=5)
    st.write("📌 Recommended Items:", recs)

# -------------------------------
# Hybrid Recommender
# -------------------------------
st.header("🔀 Hybrid Recommendation (CBF + CF)")

def hybrid_recommend(visitorid, itemid, alpha=0.6, top_n=5):
    cf_recs = recommend_cf(visitorid, top_n*2)
    cb_recs = recommend_content(itemid, top_n*2)
    scores = {}
    for i, it in enumerate(cf_recs):
        scores[it] = scores.get(it, 0) + alpha*(1/(i+1))
    for i, it in enumerate(cb_recs):
        scores[it] = scores.get(it, 0) + (1-alpha)*(1/(i+1))
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

visitorid_hybrid = st.number_input("Visitor ID (Hybrid):", min_value=0, step=1, key="hybrid_visitor")
itemid_hybrid = st.number_input("Item ID (Hybrid):", min_value=0, step=1, key="hybrid_item")
if st.button("Get Hybrid Recommendations"):
    recs = hybrid_recommend(visitorid=visitorid_hybrid, itemid=itemid_hybrid, top_n=5)
    st.write("📌 Hybrid Recommended Items:", recs)

# -------------------------------
# User Segmentation
# -------------------------------
st.header("👤 User Segmentation (K-Means)")

events_sample["timestamp"] = pd.to_datetime(events_sample["timestamp"])
events_sample = events_sample.sort_values(by=['visitorid', 'timestamp'])
events_sample["time_diff"] = events_sample.groupby("visitorid")["timestamp"].diff().dt.total_seconds().fillna(0)

user_features = events_sample.groupby("visitorid").agg(
    num_events=("event", "count"),
    unique_events=("event", lambda x: x.nunique()),
    time_spent=("time_diff", "sum"),
    avg_time_between=("time_diff", "mean"),
    num_items_viewed=("itemid", lambda x: x[events_sample.loc[x.index, 'event'] == 'view'].nunique()),
    num_adds_to_cart=("itemid", lambda x: x[events_sample.loc[x.index, 'event'] == 'addtocart'].nunique()),
    num_transactions=("itemid", lambda x: x[events_sample.loc[x.index, 'event'] == 'transaction'].nunique())
).reset_index()

scaler = StandardScaler()
scaled_user_features = scaler.fit_transform(user_features.drop("visitorid", axis=1))
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_features["cluster_label"] = kmeans.fit_predict(scaled_user_features)

st.write("📊 Distribution of Users Across Clusters:")
st.bar_chart(user_features["cluster_label"].value_counts())

# -------------------------------
# Anomaly Detection
# -------------------------------
st.header("🚨 Anomaly Detection (Isolation Forest)")

features_for_anomaly = user_features.drop(["visitorid", "cluster_label"], axis=1)
scaled_features = scaler.transform(features_for_anomaly)
isolation_forest = IsolationForest(contamination="auto", random_state=42)
user_features["anomaly_label"] = isolation_forest.fit_predict(scaled_features)

st.write("✅ Normal Users:", (user_features["anomaly_label"] == 1).sum())
st.write("⚠️ Abnormal Users:", (user_features["anomaly_label"] == -1).sum())

abnormal_visitors = user_features[user_features["anomaly_label"] == -1]["visitorid"].tolist()
if st.checkbox("Show Abnormal User IDs"):
    st.write(abnormal_visitors)
