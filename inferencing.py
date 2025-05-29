# Import Library
import streamlit as st
import pickle  # Ganti dari joblib ke pickle untuk konsistensi
import pandas as pd

# Judul Aplikasi
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="centered")
st.title('🎬 Sistem Rekomendasi Film Netflix')

# Membaca data yang sudah diproses
@st.cache_data  # Untuk caching biar lebih cepat
def load_data():
    # Baca file dengan pickle (sesuai cara penyimpanan di OOP code)
    with open('x-ipynb.pkl', 'rb') as f:
        smd = pickle.load(f)
    with open('cosine_sim-ipynb.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return smd, cosine_sim

smd, cosine_sim = load_data()

# Fungsi rekomendasi (diubah sesuai versi OOP)
def get_recommendations(title, _smd, _cosine_sim, num_recommend=5):
    try:
        # Cari index film yang sesuai
        indices = pd.Series(_smd.index, index=_smd['title']).drop_duplicates()
        idx = indices[title]
        
        # Handle jika ada duplicate title
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        # Hitung similarity score
        sim_scores = list(enumerate(_cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil rekomendasi teratas
        top_similar = sim_scores[1:num_recommend+1]
        movie_indices = [i[0] for i in top_similar]
        
        # Format output
        recommendations = _smd.iloc[movie_indices].copy()
        recommendations['Score'] = [i[1] for i in top_similar]
        return recommendations.drop(columns=['text'], errors='ignore')
        
    except KeyError:
        return None  # Kasus ketika film tidak ditemukan

# Input dari user
movie_title = st.text_input('Masukkan judul film yang Anda sukai:')

if movie_title:
    # Cari film (case insensitive)
    matched_movies = smd[smd['title'].str.lower() == movie_title.lower()]
    
    if not matched_movies.empty:
        # Ambil judul asli (untuk handle case sensitivity)
        actual_title = matched_movies.iloc[0]['title']
        st.success(f'Film "{actual_title}" ditemukan!')
        
        # Dapatkan rekomendasi
        recommendations = get_recommendations(actual_title, smd, cosine_sim)
        
        # Tampilkan rekomendasi
        st.subheader('Rekomendasi untuk Anda:')
        cols = st.columns(len(recommendations))
        
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f0f2f6; 
                        padding:15px; 
                        border-radius:10px; 
                        box-shadow:0 4px 8px rgba(0,0,0,0.05); 
                        text-align:center; 
                        height: 130px; 
                        display: flex; 
                        flex-direction: column; 
                        justify-content: center; 
                        align-items: center;">
                        <p style="font-weight:bold;color:#333;">{row['title']}</p>
                        <p style="color:gray;font-size:12px">Kesamaan: {row['Score']:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error('❌ Film tidak ditemukan. Coba cek judul atau masukkan judul lain.')