import re                                      #imported python regex
from nltk.corpus import stopwords              #imported nltk module, used for preprocessing text data. has about 200 words - a, an, or, the, etc.
from nltk.tokenize import word_tokenize        #import tokenize method - assigns a number to each word, which is a token.
from bs4 import BeautifulSoup                  #It is a html and xml parser, used for scraping web pages.
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import pickle
print("loading model...")
with open('IMDBReviews.pkl', 'rb') as f:
    IMDBReviews, vectorizer = pickle.load(f)
print("model loaded...")

def preprocess_dataset(reviews):
    result = []
    for review in reviews:
        review_text = BeautifulSoup(review, "html.parser").get_text()
        review_text = re.sub(r'[^a-zA-Z]', ' ', review_text)
        review_text = review_text.lower()
        words = word_tokenize(review_text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        preprocessed_review = ' '.join(words)
        result.append(preprocessed_review)
    return(result)

howIsThisReview = ["what a lovely movie, awesome it was. there were many casts who delivered {}{}--- extraordinary *7 performance!!"]
prod_result = preprocess_dataset(howIsThisReview)

bow_features = vectorizer.transform(prod_result)
bow_features_sparse = csr_matrix(bow_features)

predicted_sentiment = IMDBReviews.predict(bow_features_sparse)

print("Review:", howIsThisReview)
print("Predicted Sentiment:", predicted_sentiment)