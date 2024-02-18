#Preprocesses data - removed noise.
import re                                      #imported python regex
from nltk.corpus import stopwords              #imported nltk module, used for preprocessing text data. has about 200 words - a, an, or, the, etc.
from nltk.tokenize import word_tokenize        #import tokenize method - assigns a number to each word, which is a token.
from bs4 import BeautifulSoup                  #It is a html and xml parser, used for scraping web pages.
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import pandas as pd

print("preprocessing data...")
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


fp = 'C:\\ml\\data\\IMDBSentiment\\IMDB Dataset.csv'
reviews_df = pd.read_csv(fp)
print(reviews_df.shape)
reviews = reviews_df['review'].tolist()
sentiment = reviews_df['sentiment'].tolist()
result = preprocess_dataset(reviews)

print("creating features...")
#create features from the reviews in using Bag of Words.
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

#maximum features limited to 5000, to fit in memory.
vectorizer = CountVectorizer(max_features=5000)
bow_features = vectorizer.fit_transform(result)
bow_features_sparse = csr_matrix(bow_features)

print("testing model...")
#create test and train data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow_features_sparse, sentiment, test_size=0.2, random_state=42)

#Logistic regression -
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a Logistic Regression classifier
logreg_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
logreg_model.fit(X_train, y_train)

# Predict labels for the test set
y_pred = logreg_model.predict(X_test)

print("calculating accuracy...")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Save the model
print("Saving model...")
import pickle
with open('IMDBReviews.pkl', 'wb') as f:
    pickle.dump((logreg_model, vectorizer),  f)
