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
# Load Data
# -------------------------------
@st.cache_data
def load_data(events_path, item_props_path, category_tree_path, sample_percentage=0.02):
    events_df_filtered = pd.read_csv(events_path)
    item_props_filtered = pd.read_csv(item_props_path)
    category_tree = pd.read_csv(category_tree_path)

    # Sample for performance
    events_sample = events_df_filtered.sample(frac=sample_percentage, random_state=42)
    item_props_sample = item_props_filtered.sample(frac=sample_percentage, random_state=42)

    return events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree


# -------------------------------
# Recommendation Models
# -------------------------------
def build_models(events_sample, item_props_sample):
    # Content-Based
    item_features = item_props_sample.groupby("itemid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(item_features["value"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(item_features.index, index=item_features["itemid"])

    # Collaborative
    weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
    events_sample["weight"] = events_sample["event"].map(weight_map).fillna(0)

    user_item_matrix = events_sample.pivot_table(
        index="visitorid", columns="itemid", values="weight", fill_value=0
    )
    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    return item_features, cosine_sim, indices, user_item_matrix, user_sim_df


def recommend_content(itemid, item_features, cosine_sim, indices, top_n=5):
    if itemid not in indices:
        return pd.DataFrame()
    idx = indices[itemid]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    item_indices = [i[0] for i in sim_scores]
    scores = [s[1] for s in sim_scores]
    return pd.DataFrame({"itemid": item_features.iloc[item_indices]["itemid"].tolist(), "score": scores})


def recommend_cf(visitorid, events_sample, user_item_matrix, user_sim_df, top_n=5):
    if visitorid not in user_item_matrix.index:
        return pd.DataFrame()

    similar_users = user_sim_df[visitorid].sort_values(ascending=False).index[1:]
    recommendations = events_sample[events_sample["visitorid"].isin(similar_users)] \
                        .groupby("itemid")["weight"].sum().sort_values(ascending=False)

    seen_items = set(events_sample[events_sample["visitorid"] == visitorid]["itemid"])
    recommendations = recommendations[~recommendations.index.isin(seen_items)]

    return recommendations.head(top_n).reset_index().rename(columns={"weight": "score"})


def hybrid_recommend(visitorid, itemid, item_features, cosine_sim, indices,
                     events_sample, user_item_matrix, user_sim_df, alpha=0.6, top_n=5):
    cf_df = recommend_cf(visitorid, events_sample, user_item_matrix, user_sim_df, top_n*2)
    cb_df = recommend_content(itemid, item_features, cosine_sim, indices, top_n*2)

    scores = {}
    for i, row in cf_df.iterrows():
        scores[row["itemid"]] = scores.get(row["itemid"], 0) + alpha*(1/(i+1))
    for i, row in cb_df.iterrows():
        scores[row["itemid"]] = scores.get(row["itemid"], 0) + (1-alpha)*(1/(i+1))

    return pd.DataFrame(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n],
                        columns=["itemid", "score"])


def recommend_by_category(itemid, item_props_filtered, top_n=5):
    item_category_map = item_props_filtered[item_props_filtered["property"] == "categoryid"].set_index("itemid")["value"].to_dict()
    if itemid not in item_category_map:
        return pd.DataFrame()
    category = item_category_map[itemid]
    same_category_items = [k for k, v in item_category_map.items() if v == category and k != itemid]
    return pd.DataFrame({"itemid": same_category_items[:top_n]})


def hybrid_category(visitorid, itemid, item_props_filtered, *args, **kwargs):
    base_recs = hybrid_recommend(visitorid, itemid, *args, **kwargs, top_n=15)
    item_category_map = item_props_filtered[item_props_filtered["property"] == "categoryid"].set_index("itemid")["value"].to_dict()
    if itemid not in item_category_map:
        return pd.DataFrame()
    target_cat = item_category_map[itemid]
    filtered = base_recs[base_recs["itemid"].map(lambda x: item_category_map.get(x) == target_cat)]
    return filtered.head(kwargs.get("top_n", 5))


# -------------------------------
# User Segmentation & Anomaly Detection
# -------------------------------
def build_user_features(events_sample):
    events_sample = events_sample.sort_values(by=["visitorid", "timestamp"])
    events_sample["timestamp"] = pd.to_datetime(events_sample["timestamp"])
    events_sample["time_diff"] = events_sample.groupby("visitorid")["timestamp"].diff().dt.total_seconds().fillna(0)

    user_features = events_sample.groupby("visitorid").agg(
        num_events=("event", "count"),
        unique_events=("event", "nunique"),
        time_spent=("time_diff", "sum"),
        avg_time_between_events=("time_diff", "mean"),
        max_time_between_events=("time_diff", "max"),
        num_items_viewed=("itemid", lambda x: (events_sample.loc[x.index, "event"] == "view").sum()),
        num_adds_to_cart=("itemid", lambda x: (events_sample.loc[x.index, "event"] == "addtocart").sum()),
        num_transactions=("itemid", lambda x: (events_sample.loc[x.index, "event"] == "transaction").sum())
    ).reset_index()
    return user_features


def cluster_users(user_features, n_clusters=3):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(user_features.drop("visitorid", axis=1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_features["cluster_label"] = kmeans.fit_predict(scaled)
    return user_features


def detect_anomalies(user_features):
    features = user_features.drop(["visitorid", "cluster_label"], axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    iso = IsolationForest(contamination="auto", random_state=42)
    user_features["anomaly_label"] = iso.fit_predict(scaled)
    return user_features


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("🛒 E-commerce Recommendation System")

# File uploads
st.sidebar.header("Upload Data Files")
events_file = st.sidebar.file_uploader("Upload events_df_filtered.csv", type="csv")
item_props_file = st.sidebar.file_uploader("Upload item_props_filtered.csv", type="csv")
category_tree_file = st.sidebar.file_uploader("Upload category_tree.csv", type="csv")

if events_file and item_props_file and category_tree_file:
    events_df_filtered, events_sample, item_props_filtered, item_props_sample, category_tree = load_data(
        events_file, item_props_file, category_tree_file
    )

    item_features, cosine_sim, indices, user_item_matrix, user_sim_df = build_models(events_sample, item_props_sample)
    user_features = build_user_features(events_sample)
    user_features = cluster_users(user_features)
    user_features = detect_anomalies(user_features)

    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "👥 User Segmentation", "🚨 Anomaly Detection"])

    with tab1:
        st.subheader("Recommendation Modes")
        mode = st.selectbox("Choose Mode", ["Content-Based", "Collaborative", "Hybrid", "Category", "Hybrid+Category"])

        if mode == "Content-Based":
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            if st.button("Recommend"):
                recs = recommend_content(item_id, item_features, cosine_sim, indices)
                st.dataframe(recs)

        elif mode == "Collaborative":
            user_id = st.number_input("Enter Visitor ID", min_value=1, step=1)
            if st.button("Recommend"):
                recs = recommend_cf(user_id, events_sample, user_item_matrix, user_sim_df)
                st.dataframe(recs)

        elif mode == "Hybrid":
            user_id = st.number_input("Enter Visitor ID", min_value=1, step=1)
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            alpha = st.slider("Alpha (balance CF vs CBF)", 0.0, 1.0, 0.6)
            if st.button("Recommend"):
                recs = hybrid_recommend(user_id, item_id, item_features, cosine_sim, indices,
                                        events_sample, user_item_matrix, user_sim_df, alpha=alpha)
                st.dataframe(recs)

        elif mode == "Category":
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            if st.button("Recommend"):
                recs = recommend_by_category(item_id, item_props_filtered)
                st.dataframe(recs)

        elif mode == "Hybrid+Category":
            user_id = st.number_input("Enter Visitor ID", min_value=1, step=1)
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            if st.button("Recommend"):
                recs = hybrid_category(user_id, item_id, item_props_filtered,
                                       item_features, cosine_sim, indices,
                                       events_sample, user_item_matrix, user_sim_df)
                st.dataframe(recs)

    with tab2:
        st.subheader("User Segments")
        st.dataframe(user_features[["visitorid", "cluster_label"]].head(20))

    with tab3:
        st.subheader("Anomaly Detection")
        st.dataframe(user_features[["visitorid", "anomaly_label"]].head(20))
else:
    st.info("⬅️ Please upload all required CSV files to begin.")