from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import ast

nltk.download('punkt')
ps = PorterStemmer()

app = Flask(__name__)

# Load and preprocess data
movies = pd.read_csv("OneDrive\Desktop\python\machine_learning\movie_recommender_website\tmdb_5000_movies.csv")
credits = pd.read_csv("OneDrive\Desktop\python\machine_learning\movie_recommender_website\tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

def convert(obj):
    return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)]

def convert3(obj):
    L = []
    for i in ast.literal_eval(obj)[:3]:
        L.append(i['name'].replace(" ", ""))
    return L

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name'].replace(" ", "")]
    return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie = request.form['movie']
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommendations = [new_df.iloc[i[0]].title for i in movie_list]
    return render_template('index.html', movies=new_df['title'].values, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
