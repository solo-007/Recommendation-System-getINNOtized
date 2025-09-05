import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.sparse import csr_matrix

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="🛒 E-commerce Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for UI Enhancements
# -------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #fdfdfd;
        font-family: "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4 {
        color: #2C3E50;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9A7, #92FE9D);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #92FE9D, #00C9A7);
        transform: scale(1.02);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2C3E50;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "Content-Based Filtering", "Collaborative Filtering", 
     "Hybrid Recommender", "User Segmentation", "Anomaly Detection"]
)


# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(sample_percentage=0.02):
    events_url = "https://drive.google.com/uc?id=1pgf3rudMhoaOGB1MeIe41NIulxPq8okd"
    item_props_url = "https://drive.google.com/uc?id=1PveWfRDtgg5ytXdepZm2l_50n_quWbbB"
    category_tree_url = "https://drive.google.com/uc?id=1dZ1eAvcPDY3RSuPBCezW9-WSRrT5uYSy"

    events_df_filtered = pd.read_csv(events_url)
    item_props_filtered = pd.read_csv(item_props_url)
    category_tree = pd.read_csv(category_tree_url)

    events_sample = events_df_filtered.sample(frac=sample_percentage, random_state=42)
    item_props_sample = item_props_filtered.sample(frac=sample_percentage, random_state=42)

    return events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree

events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree = load_data()

# -------------------------------
# Home
# -------------------------------
if page == "🏠 Home":
    st.title("🛒 E-commerce Recommendation Dashboard")
    st.markdown("Welcome to the **E-commerce Recommender System** 🎉")
    st.info("Navigate through the sidebar to explore different recommendation techniques.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Events Loaded", len(events_df_filtered))
    col2.metric("📦 Unique Items", item_props_filtered["itemid"].nunique())
    col3.metric("👥 Users", events_df_filtered["visitorid"].nunique())

    with st.expander("📂 Preview Data"):
        st.subheader("Events Sample")
        st.dataframe(events_sample.head())
        st.subheader("Item Properties Sample")
        st.dataframe(item_props_sample.head())
        st.subheader("Category Tree")
        st.dataframe(category_tree.head())


# -------------------------------
# Global User Features
# -------------------------------
events_sample["timestamp"] = pd.to_datetime(events_sample["timestamp"], errors="coerce")
events_sample = events_sample.sort_values(by=["visitorid", "timestamp"])
events_sample["time_diff"] = events_sample.groupby("visitorid")["timestamp"].diff().dt.total_seconds().fillna(0)

user_features = events_sample.groupby("visitorid").agg(
    num_events=("event", "count"),
    unique_events=("event", pd.Series.nunique),
    time_spent=("time_diff", "sum"),
    avg_time_between=("time_diff", "mean"),
    max_time_between=("time_diff", "max")
).reset_index()

views = events_sample[events_sample["event"] == "view"]
carts = events_sample[events_sample["event"] == "addtocart"]
transactions = events_sample[events_sample["event"] == "transaction"]

user_features["num_items_viewed"] = views.groupby("visitorid")["itemid"].nunique()
user_features["num_adds_to_cart"] = carts.groupby("visitorid")["itemid"].nunique()
user_features["num_transactions"] = transactions.groupby("visitorid")["itemid"].nunique()
user_features = user_features.fillna(0)

scaler = StandardScaler()
scaled_user_features = scaler.fit_transform(user_features.drop("visitorid", axis=1))


# -------------------------------
# Shared Global Models + Functions
# -------------------------------

# --- Content-Based Filtering setup ---
item_features = item_props_sample.groupby("itemid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(item_features["value"])
nn_model_cbf = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model_cbf.fit(tfidf_matrix)
indices = pd.Series(item_features.index, index=item_features["itemid"])

def recommend_content(itemid, top_n=5):
    if itemid not in indices:
        return None
    idx = indices[itemid]
    distances, neighbors = nn_model_cbf.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    return item_features.iloc[neighbors.flatten()[1:]]["itemid"].tolist()

# --- Collaborative Filtering setup ---
weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
events_sample["weight"] = events_sample["event"].map(weight_map).fillna(0)
visitor_id_map = {vid: i for i, vid in enumerate(events_sample["visitorid"].unique())}
item_id_map = {iid: i for i, iid in enumerate(events_sample["itemid"].unique())}
events_sample["visitor_idx"] = events_sample["visitorid"].map(visitor_id_map)
events_sample["item_idx"] = events_sample["itemid"].map(item_id_map)

user_item_sparse = csr_matrix((events_sample["weight"], (events_sample["visitor_idx"], events_sample["item_idx"])))
nn_model_cf = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model_cf.fit(user_item_sparse)

def recommend_cf(visitorid, top_n=5):
    if visitorid not in visitor_id_map:
        return None
    visitor_idx = visitor_id_map[visitorid]
    distances, neighbors = nn_model_cf.kneighbors(user_item_sparse[visitor_idx], n_neighbors=top_n+1)
    neighbor_users = neighbors.flatten()[1:]
    neighbor_data = events_sample[events_sample["visitor_idx"].isin(neighbor_users)]
    recs = neighbor_data.groupby("itemid")["weight"].sum().sort_values(ascending=False)
    seen_items = set(events_sample[events_sample["visitorid"] == visitorid]["itemid"])
    return recs[~recs.index.isin(seen_items)].head(top_n).index.tolist()

# -------------------------------
# Content-Based Filtering Page
# -------------------------------
if page == "Content-Based Filtering":
    st.header("🎯 Content-Based Filtering")
    itemid_input = st.number_input("Enter an Item ID:", min_value=0, step=1)
    if st.button("Get Content-Based Filtering Recommendations"):
        recs = recommend_content(itemid=itemid_input, top_n=5)
        if recs is None:
            st.error(f"❌ Item ID {itemid_input} not found.")
        elif len(recs) > 0:
            st.success(f"✅ Recommended items for Item ID: {itemid_input}:")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.warning("⚠️ No similar items found.")

# -------------------------------
# Collaborative Filtering Page
# -------------------------------
if page == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering")
    visitorid_input = st.number_input("Enter a Visitor ID:", min_value=0, step=1)
    if st.button("Get Collaborative Filtering Recommendations"):
        recs = recommend_cf(visitorid=visitorid_input, top_n=5)
        if recs is None:
            st.error(f"❌ Visitor ID {visitorid_input} not found.")
        elif len(recs) > 0:
            st.success(f"✅ Recommended items for Visitor ID: {visitorid_input}:")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.warning("⚠️ No similar Visitor found.")

# -------------------------------
# Hybrid Page
# -------------------------------
if page == "Hybrid Recommender":
    st.header("🔀 Hybrid Recommendations")

    def hybrid_recommend(visitorid, itemid, alpha=0.6, top_n=5):
        cf_recs = recommend_cf(visitorid, top_n*2)
        cb_recs = recommend_content(itemid, top_n*2)
        if cf_recs is None or cb_recs is None:
            return None
        scores = {}
        for i, it in enumerate(cf_recs):
            scores[it] = scores.get(it, 0) + alpha*(1/(i+1))
        for i, it in enumerate(cb_recs):
            scores[it] = scores.get(it, 0) + (1-alpha)*(1/(i+1))
        return sorted(scores, key=scores.get, reverse=True)[:top_n]

    visitorid_hybrid = st.number_input("Visitor ID:", min_value=0, step=1)
    itemid_hybrid = st.number_input("Item ID:", min_value=0, step=1)
    if st.button("Get Hybrid Recommendations"):
        recs = hybrid_recommend(visitorid=visitorid_hybrid, itemid=itemid_hybrid, top_n=5)
        if recs is None:
            st.error("❌ Invalid Visitor ID or Item ID")
        elif len(recs) > 0:
            st.success(f"✅ Hybrid Recommendations:")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.warning("⚠️ Either No similar Visitor or Item found.")


# -------------------------------
# User Segmentation
# -------------------------------
if page == "User Segmentation":
    st.header("👤 User Segmentation (K-Means)")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    user_features["cluster_label"] = kmeans.fit_predict(scaled_user_features)
    st.bar_chart(user_features["cluster_label"].value_counts())

# -------------------------------
# Anomaly Detection
# -------------------------------
if page == "Anomaly Detection":
    st.header("🚨 Anomaly Detection (Isolation Forest)")
    features_for_anomaly = user_features.drop(["visitorid"], axis=1)
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    user_features["anomaly_label"] = isolation_forest.fit_predict(scaled_user_features)
    st.write("✅ Normal Users:", (user_features["anomaly_label"] == 1).sum())
    st.write("⚠️ Abnormal Users:", (user_features["anomaly_label"] == -1).sum())
    if st.checkbox("Show Abnormal User IDs"):
        st.write(user_features[user_features["anomaly_label"] == -1]["visitorid"].tolist())

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <div class="footer">
        Developed By <b>Solomon Sannie</b> | Supervised by <b>Precious Darkwa</b> | <b>AZUBI AFRICA</b>
    </div>
    """,
    unsafe_allow_html=True
)
