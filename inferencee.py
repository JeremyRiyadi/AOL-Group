import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Netflix Recommendation System", layout="centered")
st.markdown(
    "<h1 style='white-space: nowrap; margin-right: 20px;'>🎬 Netflix Recommendation System</h1>",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    with open('x.pkl', 'rb') as f:
        smd = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return smd, tfidf

smd, tfidf = load_data()
cosine_sim = linear_kernel(tfidf, tfidf)

def get_recommendations(title, _smd, _cosine_sim, num_recommend=5):
    try:
        indices = pd.Series(_smd.index, index=_smd['title']).drop_duplicates()
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        sim_scores = list(enumerate(_cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar = sim_scores[1:num_recommend+1]
        movie_indices = [i[0] for i in top_similar]
        recommendations = _smd.iloc[movie_indices].copy()
        recommendations['Score'] = [i[1] for i in top_similar]
        return recommendations.drop(columns=['text'], errors='ignore')
    except KeyError:
        return None

selected_title = st.selectbox(
    "🎬 Search and select a movie title:",
    options=smd['title'].dropna().unique().tolist(),
    index=None,
    placeholder="Start typing a movie title..."
)

if selected_title:
    st.success(f'🎉 Movie "{selected_title}" found successfully! Enjoy the show 🍿')
    recommendations = get_recommendations(selected_title, smd, cosine_sim)

    if recommendations is not None:
        st.subheader("✨ Personalized Recommendations Just for You! 💡")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #e0e7ff, #fdf2f8);
                    padding: 24px;
                    border-radius: 16px;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.08);
                    max-width: 600px;
                    margin: auto;
                    font-family: 'Segoe UI', sans-serif;
                    color: #1f2937;
                    margin-bottom: 30px;
                    ">
                    <h2 style="margin-bottom: 16px; font-size: 24px; font-weight: 700; color: #111827;">
                        🎬 Recommendation for You : <span style="color:#4f46e5">{row['title']}</span>
                    </h2>
                    <ul style="list-style: none; padding: 0; font-size: 14px;">
                        <li><strong>Director :</strong>  {row['director']}</li>
                        <li><strong>Cast :</strong> {row['cast']}</li>
                        <li><strong>Country :</strong> {row['country']}</li>
                        <li><strong>Date Added :</strong> {row['date_added']}</li>
                        <li><strong>Release Year :</strong> {row['release_year']}</li>
                        <li><strong>Rating :</strong> {row['rating']}</li>
                        <li><strong>Duration :</strong> {row['listed_in']}</li>
                        <li><strong>Genre :</strong> Documentaries</li>
                    </ul>
                    <p style="margin-top: 16px; line-height: 1.6; font-size: 14px; background: #fefce8; padding: 12px; border-radius: 8px;">
                        <em><strong>Description :</strong> <br/>{row['description']}</em>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Sorry, Movie Not Found. Please Check Your Spelling and Try Again")