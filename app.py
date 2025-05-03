import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from newspaper import Article, ArticleException # For web scraping [5]
import logging

# --- Configuration ---
MODEL_DIR = 'models'
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_binary_best.keras')
MULTI_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_multi_best.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer_lstm.pkl')

# Load from your notebook [1]
MAX_SEQUENCE_LENGTH_LSTM = 256
# Define your class labels based on training (ensure order matches model output)
# Assuming 0: business, 1: political, 2: scientific, 3: spiritual, 4: sports
# Load from label_encoder if saved, otherwise hardcode
MULTI_CLASS_LABELS = ['Business', 'Political', 'Scientific', 'Spiritual', 'Sports']
BINARY_CLASS_LABELS = {0: 'Fake', 1: 'Real'} # Based on notebook [1] and search result [2]

# --- Initialize Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load Models and Tokenizer ---
try:
    app.logger.info("Loading Keras models...")
    lstm_model_binary = tf.keras.models.load_model(BINARY_MODEL_PATH)
    lstm_model_multi = tf.keras.models.load_model(MULTI_MODEL_PATH)
    app.logger.info("Models loaded successfully.")

    app.logger.info("Loading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer_lstm = pickle.load(handle)
    app.logger.info("Tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading models or tokenizer: {e}")
    # Handle the error appropriately (e.g., exit or run without ML features)
    lstm_model_binary = None
    lstm_model_multi = None
    tokenizer_lstm = None

from newspaper import Article, ArticleException, Config # Import Config
import logging # Assuming you have logging setup as 'app.logger'
# --- Helper Functions ---
def scrape_article(url):
    """Scrapes headline and text content from a given URL using newspaper3k with custom config."""
    try:
        # Configure newspaper3k
        config = Config()
        config.request_timeout = 20 # Set timeout to 20 seconds (adjust as needed)
        # Optionally, set a user agent (good practice)
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        config.memoize_articles = False # Avoid caching issues during testing/debugging
        
        article = Article(url, config=config) # Pass config to Article
        article.download()
        article.parse()
        
        content = f"{article.title}. {article.text}"
        if not content or content.strip() == '.':
            app.logger.warning(f"No content extracted from {url}")
            return None, "Could not extract significant content from the URL."
        return content, None 

    except ArticleException as e:
        app.logger.error(f"Newspaper3k scraping failed for {url}: {e}")
        # Check if the error is specifically a timeout
        if 'Read timed out' in str(e):
            return None, f"Failed to scrape: The connection timed out after {config.request_timeout} seconds."
        else:
            return None, f"Failed to scrape the article. Error: {e}"
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during scraping {url}: {e}")
        return None, f"An unexpected error occurred: {e}"


def preprocess_text(text):
    """Preprocesses text using the loaded tokenizer and padding."""
    if not tokenizer_lstm:
        raise ValueError("Tokenizer not loaded.")
    sequences = tokenizer_lstm.texts_to_sequences([text]) # Input must be a list
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH_LSTM)
    return padded_sequence

def predict_news(text):
    """Makes predictions using the loaded LSTM models."""
    if not lstm_model_binary or not lstm_model_multi:
        raise ValueError("Models not loaded.")

    processed_text = preprocess_text(text)

    # Binary Prediction (Real/Fake)
    binary_pred_prob = lstm_model_binary.predict(processed_text)[0][0]
    binary_pred_label_idx = int(binary_pred_prob > 0.5) # Threshold at 0.5
    binary_prediction = BINARY_CLASS_LABELS.get(binary_pred_label_idx, "Unknown")
    binary_confidence = binary_pred_prob if binary_pred_label_idx == 1 else 1 - binary_pred_prob

    # Multiclass Prediction (Category)
    multi_pred_probs = lstm_model_multi.predict(processed_text)[0]
    multi_pred_label_idx = np.argmax(multi_pred_probs)
    multi_prediction = MULTI_CLASS_LABELS[multi_pred_label_idx] if multi_pred_label_idx < len(MULTI_CLASS_LABELS) else "Unknown Category"
    multi_confidence = multi_pred_probs[multi_pred_label_idx]

    return {
        'binary_prediction': binary_prediction,
        'binary_confidence': f"{binary_confidence:.2f}",
        'multi_prediction': multi_prediction,
        'multi_confidence': f"{multi_confidence:.2f}"
    }

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles URL submission, scraping, preprocessing, and prediction."""
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return render_template('index.html', error="Please provide a URL.")

        if not lstm_model_binary or not lstm_model_multi or not tokenizer_lstm:
             return render_template('index.html', error="Models or Tokenizer not loaded. Cannot perform prediction.")

        app.logger.info(f"Received URL: {url}")

        # Scrape the article
        scraped_content, scrape_error = scrape_article(url)

        if scrape_error:
            app.logger.error(f"Scraping failed for {url}: {scrape_error}")
            return render_template('index.html', url=url, error=scrape_error)

        if not scraped_content:
             app.logger.error(f"No content scraped from {url}")
             return render_template('index.html', url=url, error="Could not scrape any content from the provided URL.")

        app.logger.info(f"Scraped content length: {len(scraped_content)}")

        # Predict using the models
        try:
            predictions = predict_news(scraped_content)
            app.logger.info(f"Predictions for {url}: {predictions}")
            return render_template('result.html',
                                   url=url,
                                   binary_prediction=predictions['binary_prediction'],
                                   binary_confidence=predictions['binary_confidence'],
                                   multi_prediction=predictions['multi_prediction'],
                                   multi_confidence=predictions['multi_confidence'],
                                   scraped_text=scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content # Show snippet
                                  )
        except Exception as e:
            app.logger.error(f"Prediction failed for {url}: {e}")
            return render_template('index.html', url=url, error=f"Prediction failed: {e}")

    # Redirect GET requests back to index
    return redirect(url_for('index'))


# --- Run the App ---
if __name__ == '__main__':
    # Use debug=False in a production environment
    app.run(debug=True)
