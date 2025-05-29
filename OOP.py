# Class Group 1 
# Members :
# - Natasha Kayla Cahyadi - 2702235891
# - Jeremy Djohar Riyadi - 2702219572
# Class : LC09 - Model Deployment

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl

class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.smd = None
        self.categorical_columns = None
        self.numerical_columns = None

    def read_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def filter_movie(self):
        self.df = self.df[self.df['type'] == 'Movie']
    
    def drop_column(self):
        self.df.drop(columns=['type'], inplace=True)

    def drop_identifier(self):
        self.df.drop(columns=['show_id'], inplace=True)

    def handle_missing_values(self):
        self.df.fillna('Unknown', inplace=True)

    def convert_to_datetime(self):
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')

    def change_to_nr(self):
        self.df.loc[self.df['rating'].isin(['UR', 'Unknown']), 'rating'] = 'NR'

    def delete_anomalies(self):
        self.df = self.df[~self.df['rating'].isin(['74 min', '66 min', '84 min'])]

    def create_working_copy(self):
        self.smd = self.df.copy()
        return self.smd

class Modeling:
    def __init__(self, smd):
        self.smd = smd
        self.x = None
        self.tfidf_matrix = None
        self.cosine_sim = None

        self.model =TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.8,
                    sublinear_tf=True)
        
    def build_text_column(self):
        self.smd['text'] = self.smd['title'] + ', ' + self.smd['director'] * 2 + ', ' + self.smd['cast'] * 3 + ', ' + self.smd['country'] + ', ' +  self.smd['rating'] + ', ' + self.smd['listed_in'] * 5 + ', ' + self.smd['description'] * 2
        self.smd.dropna(subset=['text'], inplace=True)
        self.smd['text'] = self.smd['text'].str.lower()

    def build_smd_pickle(self):
        pkl.dump(self.smd, open('smd.pkl', 'wb'))
    
    def duplicate_text_column(self):
        self.x = self.smd.copy()
        self.x = self.x.drop_duplicates(subset=['text']).reset_index(drop=True)

    def fit_model(self):
        self.tfidf_matrix = self.model.fit_transform(self.x['text']).toarray()
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def build_model_pickle(self):
        pkl.dump(self.model, open('tfidf.pkl', 'wb'))

    def build_cosine_sim_pickle(self):
        pkl.dump(self.cosine_sim, open('cosine_sim.pkl', 'wb'))

    def get_recommendations(self, title, num_recommend=5):
        try:
            indices = pd.Series(self.x.index, index=self.x['title']).drop_duplicates()
            idx = indices[title]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_similar = sim_scores[1:num_recommend+1]
            movie_indices = [i[0] for i in top_similar]
            ret_smd = self.smd.iloc[movie_indices].copy()
            ret_smd['Score'] = [i[1] for i in top_similar]
            return ret_smd.drop(columns=['text'], errors='ignore')
        except KeyError:
            print(f"{title} not found. Please check the title or try another.")
            return pd.DataFrame()
        
preprocessor = Preprocessor('netflix_titles.csv')
preprocessor.read_data()
preprocessor.filter_movie()
preprocessor.drop_column()
preprocessor.drop_identifier()
preprocessor.handle_missing_values()
preprocessor.convert_to_datetime()
preprocessor.change_to_nr()
preprocessor.delete_anomalies()
preprocessor.create_working_copy()

modeling = Modeling(preprocessor.smd)
modeling.build_text_column()
modeling.build_smd_pickle()
modeling.duplicate_text_column()
modeling.fit_model()
modeling.build_model_pickle()
modeling.build_cosine_sim_pickle()

recommendations = modeling.get_recommendations('Naruto Shippuden: The Movie')
print(recommendations)