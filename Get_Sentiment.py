from flask import Flask, request, jsonify
import pickle
from scipy.sparse import csr_matrix
import Preprocess_Dataset

app = Flask(__name__)

# Load the model
with open('IMDBReviews.pkl', 'rb') as f:
    IMDBReviews, vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data['input'])
    # Assuming your model expects a certain format of input data
    prod_result = Preprocess_Dataset.preprocess_dataset(data['input'])
    bow_features = vectorizer.transform(prod_result)
    bow_features_sparse = csr_matrix(bow_features)

    prediction = IMDBReviews.predict(bow_features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
