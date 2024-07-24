import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import requests
import pandas as pd
import torch

class LicitacionesModel:
    def __init__(self):
        self.url = "https://api.mercadopublico.cl/servicios/v1/publico/licitaciones.json?ticket=2E87FBAF-2AF6-4AD9-81D8-178A9E00D963"
        self.nlp = spacy.load("es_core_news_sm")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.licitaciones = self.fetch_licitaciones()

    def fetch_licitaciones(self):
        response = requests.get(self.url)
        data = response.json()
        licitaciones = pd.DataFrame(data['Listado'])
        licitaciones['Nombre'] = licitaciones['Nombre'].replace('', float('NaN'))
        licitaciones = licitaciones.dropna(subset=['Nombre'])
        licitaciones['Nombre_preprocessed'] = licitaciones['Nombre'].apply(self.preprocess_text)
        licitaciones['Nombre_preprocessed'] = licitaciones['Nombre_preprocessed'].replace('', float('NaN'))
        licitaciones = licitaciones.dropna(subset=['Nombre_preprocessed'])
        return licitaciones

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def calculate_similarity(self, embedding, preference_embeddings):
        return max(cosine_similarity([embedding], preference_embeddings).flatten())

    def find_compatible_licitations(self, preferences):
        preferences_preprocessed = [self.preprocess_text(pref) for pref in preferences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.licitaciones['Nombre_preprocessed'])
        preference_vector = vectorizer.transform(preferences_preprocessed)
        similarity_scores = cosine_similarity(tfidf_matrix, preference_vector).max(axis=1)
        self.licitaciones['Similarity_TFIDF'] = similarity_scores
        self.licitaciones['BERT_Embedding'] = self.licitaciones['Nombre_preprocessed'].apply(lambda x: self.get_bert_embedding(x).flatten())
        preference_embeddings = [self.get_bert_embedding(pref).flatten() for pref in preferences_preprocessed]
        self.licitaciones['Similarity_BERT'] = self.licitaciones['BERT_Embedding'].apply(lambda x: self.calculate_similarity(x, preference_embeddings))
        self.licitaciones['Final_Score'] = self.licitaciones['Similarity_TFIDF'] * 0.5 + self.licitaciones['Similarity_BERT'] * 0.5
        top_n = 10
        licitaciones_relevantes = self.licitaciones.nlargest(top_n, 'Final_Score')
        return licitaciones_relevantes[['Nombre']].to_dict(orient='records')
