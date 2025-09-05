import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(events_path, events_filtered_path, item_props_path, category_tree_path, sample_percentage=0.02):
    try:
        events = pd.read_csv(events_path)
        category_tree = pd.read_csv(category_tree_path)
        events_df_filtered = pd.read_csv(events_filtered_path)
        item_props_filtered = pd.read_csv(item_props_path)

        # Sample for performance
        item_props_sample = item_props_filtered.sample(frac=sample_percentage, random_state=42)
        events_sample = events_df_filtered.sample(frac=sample_percentage, random_state=42)

        return events, category_tree, events_sample, item_props_sample
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


# -------------------------------
# Build Models
# -------------------------------
def build_models(events_sample, item_props_sample):
    # ---- Content-Based ----
    item_features = item_props_sample.groupby("itemid")["value"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(item_features["value"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(item_features.index, index=item_features["itemid"])

    # ---- Collaborative Filtering ----
    weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
    events_sample["weight"] = events_sample["event"].map(weight_map).fillna(0)

    user_item_matrix = events_sample.pivot_table(
        index="visitorid", columns="itemid", values="weight", fill_value=0
    )
    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    return item_features, cosine_sim, indices, user_item_matrix, user_sim_df


# -------------------------------
# Recommenders
# -------------------------------
def recommend_content(itemid, item_features, cosine_sim, indices, top_n=5):
    if itemid not in indices:
        return pd.DataFrame()
    idx = indices[itemid]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
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

    hybrid_df = pd.DataFrame(scores.items(), columns=["itemid", "score"])
    hybrid_df = hybrid_df.sort_values("score", ascending=False).head(top_n)
    return hybrid_df


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("🛒 E-commerce Recommendation System")

st.sidebar.header("Upload CSV Data")
events_file = st.sidebar.file_uploader("Upload events.csv", type="csv")
events_filtered_file = st.sidebar.file_uploader("Upload events_df_filtered.csv", type="csv")
item_props_file = st.sidebar.file_uploader("Upload item_props_filtered.csv", type="csv")
category_tree_file = st.sidebar.file_uploader("Upload category_tree.csv", type="csv")

if events_file and events_filtered_file and item_props_file and category_tree_file:
    events, category_tree, events_sample, item_props_sample = load_data(
        events_file, events_filtered_file, item_props_file, category_tree_file
    )

    if events is not None:
        st.success("✅ Data loaded and sampled")

        item_features, cosine_sim, indices, user_item_matrix, user_sim_df = build_models(events_sample, item_props_sample)

        mode = st.radio("Select Recommendation Mode", ["Content-Based", "Collaborative", "Hybrid"])

        if mode == "Content-Based":
            st.subheader("🔍 Content-Based Recommendations")
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            top_n = st.slider("Number of Recommendations", 1, 10, 5)
            if st.button("Recommend (Content-Based)"):
                recs = recommend_content(item_id, item_features, cosine_sim, indices, top_n)
                if not recs.empty:
                    st.bar_chart(recs.set_index("itemid"))
                else:
                    st.warning("No recommendations found for this Item ID")

        elif mode == "Collaborative":
            st.subheader("👥 Collaborative Filtering Recommendations")
            user_id = st.number_input("Enter Visitor ID", min_value=1, step=1)
            top_n = st.slider("Number of Recommendations", 1, 10, 5)
            if st.button("Recommend (Collaborative)"):
                recs = recommend_cf(user_id, events_sample, user_item_matrix, user_sim_df, top_n)
                if not recs.empty:
                    st.bar_chart(recs.set_index("itemid"))
                else:
                    st.warning("No recommendations found for this Visitor ID")

        elif mode == "Hybrid":
            st.subheader("⚡ Hybrid Recommendations")
            user_id = st.number_input("Enter Visitor ID", min_value=1, step=1)
            item_id = st.number_input("Enter Item ID", min_value=1, step=1)
            alpha = st.slider("Weight CF vs CBF", 0.0, 1.0, 0.6)
            top_n = st.slider("Number of Recommendations", 1, 10, 5)
            if st.button("Recommend (Hybrid)"):
                recs = hybrid_recommend(user_id, item_id, item_features, cosine_sim, indices,
                                        events_sample, user_item_matrix, user_sim_df,
                                        alpha=alpha, top_n=top_n)
                if not recs.empty:
                    st.bar_chart(recs.set_index("itemid"))
                else:
                    st.warning("No hybrid recommendations found")
else:
    st.info("⬅️ Upload all required CSVs to begin.")