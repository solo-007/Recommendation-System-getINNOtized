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
# Content-Based Filtering (CBF) with NearestNeighbors
# -------------------------------
st.header("🎯 Content-Based Filtering (CBF)")

from sklearn.neighbors import NearestNeighbors

# Build item profiles (combine all property values into a text string per item)
item_features = (
    item_props_sample.groupby("itemid")["value"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(item_features["value"])

# Use Nearest Neighbors instead of full similarity matrix
nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model.fit(tfidf_matrix)

# Map item_id to index
indices = pd.Series(item_features.index, index=item_features["itemid"])

def recommend_content(itemid, top_n=5):
    """Return top_n most similar items for a given itemid"""
    if itemid not in indices:
        return []
    idx = indices[itemid]
    distances, neighbors = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    neighbor_ids = neighbors.flatten()[1:]  # skip the item itself
    return item_features.iloc[neighbor_ids]["itemid"].tolist()

# Streamlit UI
itemid_input = st.number_input("Enter an Item ID:", min_value=0, step=1, key="cbf")
if st.button("Get CBF Recommendations"):
    recs = recommend_content(itemid=itemid_input, top_n=5)
    st.write("📌 Recommended Items:", recs)


# -------------------------------
# Collaborative Filtering (Optimized with Sparse Matrix + NearestNeighbors)
# -------------------------------
st.header("👥 Collaborative Filtering (CF)")

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Map implicit events to weights
weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
events_sample["weight"] = events_sample["event"].map(weight_map).fillna(0)

# Reindex visitorid and itemid to avoid huge sparse matrices with gaps
visitor_id_map = {vid: i for i, vid in enumerate(events_sample["visitorid"].unique())}
item_id_map = {iid: i for i, iid in enumerate(events_sample["itemid"].unique())}

events_sample["visitor_idx"] = events_sample["visitorid"].map(visitor_id_map)
events_sample["item_idx"] = events_sample["itemid"].map(item_id_map)

# Create sparse user–item matrix
user_item_sparse = csr_matrix(
    (events_sample["weight"], (events_sample["visitor_idx"], events_sample["item_idx"]))
)

st.write("✅ Sparse user–item matrix shape:", user_item_sparse.shape)

# Fit NearestNeighbors on sparse matrix
nn_model_cf = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model_cf.fit(user_item_sparse)

def recommend_cf(visitorid, top_n=5):
    """Recommend items for a visitor using CF"""
    if visitorid not in visitor_id_map:
        return []
    
    visitor_idx = visitor_id_map[visitorid]
    distances, neighbors = nn_model_cf.kneighbors(
        user_item_sparse[visitor_idx], n_neighbors=top_n+1
    )
    
    neighbor_users = neighbors.flatten()[1:]  # skip self
    
    # Aggregate items from similar users
    neighbor_data = events_sample[events_sample["visitor_idx"].isin(neighbor_users)]
    recommendations = (
        neighbor_data.groupby("itemid")["weight"].sum().sort_values(ascending=False)
    )
    
    # Exclude items the visitor already interacted with
    seen_items = set(events_sample[events_sample["visitorid"] == visitorid]["itemid"])
    recommendations = recommendations[~recommendations.index.isin(seen_items)]
    
    return recommendations.head(top_n).index.tolist()

# Streamlit UI
visitorid_input = st.number_input("Enter a Visitor ID:", min_value=0, step=1, key="cf")
if st.button("Get CF Recommendations"):
    recs = recommend_cf(visitorid=visitorid_input, top_n=5)
    if recs:
        st.write("📌 Recommended Items:", recs)
    else:
        st.warning("No recommendations found (visitor may not exist or has too few interactions).")


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
# User Segmentation (Optimized)
# -------------------------------
st.header("👤 User Segmentation (K-Means)")

# Ensure datetime conversion
events_sample["timestamp"] = pd.to_datetime(events_sample["timestamp"], errors="coerce")

# Sort and compute time differences
events_sample = events_sample.sort_values(by=["visitorid", "timestamp"])
events_sample["time_diff"] = (
    events_sample.groupby("visitorid")["timestamp"].diff().dt.total_seconds().fillna(0)
)

# Precompute masks for efficiency
views = events_sample[events_sample["event"] == "view"]
carts = events_sample[events_sample["event"] == "addtocart"]
transactions = events_sample[events_sample["event"] == "transaction"]

# Aggregate user features
user_features = events_sample.groupby("visitorid").agg(
    num_events=("event", "count"),
    unique_events=("event", pd.Series.nunique),
    time_spent=("time_diff", "sum"),
    avg_time_between=("time_diff", "mean"),
    max_time_between=("time_diff", "max")
).reset_index()

# Add counts without scanning full dataframe repeatedly
user_features["num_items_viewed"] = views.groupby("visitorid")["itemid"].nunique()
user_features["num_adds_to_cart"] = carts.groupby("visitorid")["itemid"].nunique()
user_features["num_transactions"] = transactions.groupby("visitorid")["itemid"].nunique()

# Replace NaNs (for users missing some event types)
user_features = user_features.fillna(0)

# Scale and cluster
scaler = StandardScaler()
scaled_user_features = scaler.fit_transform(user_features.drop("visitorid", axis=1))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_features["cluster_label"] = kmeans.fit_predict(scaled_user_features)

st.write("📊 Distribution of Users Across Clusters:")
st.bar_chart(user_features["cluster_label"].value_counts())

# -------------------------------
# Anomaly Detection (Optimized)
# -------------------------------
st.header("🚨 Anomaly Detection (Isolation Forest)")

features_for_anomaly = user_features.drop(["visitorid", "cluster_label"], axis=1)

# Already scaled above, reuse
scaled_features = scaler.transform(features_for_anomaly)

isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # limit anomalies
user_features["anomaly_label"] = isolation_forest.fit_predict(scaled_features)

st.write("✅ Normal Users:", (user_features["anomaly_label"] == 1).sum())
st.write("⚠️ Abnormal Users:", (user_features["anomaly_label"] == -1).sum())

if st.checkbox("Show Abnormal User IDs"):
    st.write(user_features[user_features["anomaly_label"] == -1]["visitorid"].tolist())
