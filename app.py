from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from flask_wtf.csrf import CSRFProtect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds
import json
from datetime import datetime
import implicit
from scipy.sparse import csr_matrix
import logging
import os
from threadpoolctl import threadpool_limits
from scipy.sparse import csr_matrix, csc_matrix

os.environ['OPENBLAS_NUM_THREADS'] = '1'

app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = 'secure'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRecommendationEngine:
    def __init__(self, products, user_interactions):
        self.products = products
        self.user_interactions = user_interactions
        self.content_based_matrix = None
        self.user_item_matrix = None
        self.implicit_model = None
        self.category_weights = None
        self.price_similarity = None
        self._initialize_matrices()
        
    def _initialize_matrices(self):
        """Initialize all recommendation matrices and models"""
        self._build_content_based_matrix()
        self._build_collaborative_matrix()
        self._build_implicit_model()
        self._calculate_category_weights()
        self._calculate_price_similarity()

    def _build_content_based_matrix(self):
        """Enhanced content-based filtering with weighted features"""
        # Create separate TF-IDF matrices for different features
        product_texts = []
        product_categories = []
        product_tags = []
        
        for p in self.products.values():
            # Combine name and description with more weight on name
            product_texts.append(f"{p['name']} {p['name']} {p['description']}")
            product_categories.append(p['category'])
            product_tags.append(p['tags'])
        
        # Create TF-IDF vectors for each feature
        tfidf_text = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_category = TfidfVectorizer()
        tfidf_tags = TfidfVectorizer(token_pattern=r'[^\s]+')
        
        text_matrix = tfidf_text.fit_transform(product_texts)
        category_matrix = tfidf_category.fit_transform(product_categories)
        tags_matrix = tfidf_tags.fit_transform(product_tags)
        
        # Combine matrices with weights
        self.content_based_matrix = np.hstack([
            0.5 * text_matrix.toarray(),
            0.3 * category_matrix.toarray(),
            0.2 * tags_matrix.toarray()
        ])

    def _build_collaborative_matrix(self):
        """Build enhanced user-item interaction matrix"""
        try:
            n_users = len(self.user_interactions)
            n_items = len(self.products)
            
            # Initialize sparse matrix in COO format for efficient construction
            rows = []
            cols = []
            data = []
            
            # Weight different types of interactions
            view_weight = 1
            cart_weight = 2
            purchase_weight = 3
            rating_weight = 4
            
            for user_idx, (user_id, interactions) in enumerate(self.user_interactions.items()):
                # Add viewed interactions
                for item_id in interactions['viewed']:
                    if item_id <= n_items:
                        rows.append(user_idx)
                        cols.append(item_id-1)
                        data.append(view_weight)
                
                # Add cart interactions
                for item_id in interactions['cart']:
                    if item_id <= n_items:
                        rows.append(user_idx)
                        cols.append(item_id-1)
                        data.append(cart_weight)
                
                # Add purchase interactions
                for item_id in interactions['purchased']:
                    if item_id <= n_items:
                        rows.append(user_idx)
                        cols.append(item_id-1)
                        data.append(purchase_weight)
                
                # Add rating interactions
                for item_id, rating in interactions['ratings'].items():
                    if item_id <= n_items:
                        rows.append(user_idx)
                        cols.append(item_id-1)
                        data.append(rating * rating_weight)
            
            # Create CSR matrix directly
            self.user_item_matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(n_users, n_items)
            )
            
            logging.debug(f"Built collaborative matrix in CSR format: {self.user_item_matrix.shape}")
            
        except Exception as e:
            logging.error(f"Error building collaborative matrix: {e}")
            raise

    def _build_implicit_model(self):
        """Initialize and train implicit feedback model"""
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                # Convert to CSR format explicitly
                if isinstance(self.user_item_matrix, csc_matrix):
                    logging.debug("Converting CSC matrix to CSR format")
                    matrix = csr_matrix(self.user_item_matrix)
                elif isinstance(self.user_item_matrix, np.ndarray):
                    logging.debug("Converting numpy array to CSR format")
                    matrix = csr_matrix(self.user_item_matrix)
                elif isinstance(self.user_item_matrix, csr_matrix):
                    logging.debug("Matrix already in CSR format")
                    matrix = self.user_item_matrix
                else:
                    logging.warning(f"Unexpected matrix type: {type(self.user_item_matrix)}")
                    matrix = csr_matrix(self.user_item_matrix)

                # Initialize ALS model
                self.implicit_model = implicit.als.AlternatingLeastSquares(
                    factors=50,
                    regularization=0.1,
                    iterations=50
                )
                
                # Ensure matrix is transposed correctly
                logging.debug("Training implicit model")
                self.implicit_model.fit(matrix.T)
                
                logging.info("Implicit model training completed successfully")
                    
        except Exception as e:
            logging.error(f"Error building implicit model: {e}")
            raise
            
    def _calculate_category_weights(self):
        """Calculate category importance weights based on user behavior"""
        category_interactions = {}
        
        for user_data in self.user_interactions.values():
            for item_id in user_data['purchased']:
                if item_id in self.products:
                    category = self.products[item_id]['category']
                    category_interactions[category] = category_interactions.get(category, 0) + 1
        
        total_interactions = sum(category_interactions.values())
        self.category_weights = {
            cat: count/total_interactions 
            for cat, count in category_interactions.items()
        }

    def _calculate_price_similarity(self):
        """Calculate price-based similarity matrix"""
        prices = np.array([p['price'] for p in self.products.values()])
        prices_2d = prices.reshape(-1, 1)
        
        # Normalize prices
        scaler = MinMaxScaler()
        normalized_prices = scaler.fit_transform(prices_2d)
        
        # Calculate price similarity using Gaussian kernel
        self.price_similarity = np.exp(-0.5 * np.square(normalized_prices - normalized_prices.T))

    def get_enhanced_content_recommendations(self, product_id, n_recommendations=5):
        """Get content-based recommendations with multiple similarity factors"""
        product_idx = product_id - 1
        
        # Calculate content similarity
        content_similarity = cosine_similarity(
            [self.content_based_matrix[product_idx]], 
            self.content_based_matrix
        ).flatten()
        
        # Get product category weight
        category = self.products[product_id]['category']
        category_weight = self.category_weights.get(category, 0.5)
        
        # Combine similarities with weights
        final_similarity = (
            0.6 * content_similarity +
            0.2 * self.price_similarity[product_idx] +
            0.2 * category_weight
        )
        
        # Get top recommendations
        similar_indices = final_similarity.argsort()[::-1][1:n_recommendations+1]
        return [self.products[idx + 1] for idx in similar_indices]

    def get_enhanced_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get collaborative recommendations using matrix factorization"""
        try:
            if user_id not in self.user_interactions:
                logging.warning(f"No interactions found for user {user_id}")
                return []
                
            user_idx = list(self.user_interactions.keys()).index(user_id)
            
            # Get user items and validate dimensions
            user_items = self.user_item_matrix[user_idx]
            n_items = self.user_item_matrix.shape[1]
            
            # Adjust n_recommendations if larger than available items
            n_recommendations = min(n_recommendations, n_items)
            
            if n_recommendations <= 0:
                logging.warning("No items available for recommendations")
                return []
                
            # Ensure CSR format
            if not isinstance(user_items, csr_matrix):
                user_items = csr_matrix(user_items)
            
            # Get recommendations with validated size
            recommendations = self.implicit_model.recommend(
                user_idx,
                user_items,
                N=n_recommendations,
                filter_already_liked_items=True
            )
            
            # Validate and return results
            valid_recommendations = [
                (item_id, score) for item_id, score in recommendations 
                if item_id < n_items
            ][:n_recommendations]
            
            return [self.products[item_id + 1] for item_id, _ in valid_recommendations]
            
        except Exception as e:
            logging.error(f"Error generating collaborative recommendations: {e}")
            return []

    def get_enhanced_hybrid_recommendations(self, user_id, product_id, n_recommendations=5):
        """Enhanced hybrid recommendations with dynamic weighting"""
        # Get both types of recommendations
        content_recs = self.get_enhanced_content_recommendations(
            product_id, 
            n_recommendations
        )
        collab_recs = self.get_enhanced_collaborative_recommendations(
            user_id, 
            n_recommendations
        )
        
        # Calculate user engagement level
        user_data = self.user_interactions.get(user_id, {})
        engagement_score = len(user_data.get('purchased', [])) + \
                        0.5 * len(user_data.get('cart', [])) + \
                        0.2 * len(user_data.get('viewed', []))
        
        # Adjust weights based on user engagement
        if engagement_score > 10:
            collab_weight = 0.7
            content_weight = 0.3
        else:
            collab_weight = 0.3
            content_weight = 0.7
        
        # Combine recommendations with weights
        hybrid_recommendations = []
        seen_products = set()
        
        for i in range(max(len(content_recs), len(collab_recs))):
            if i < len(content_recs):
                product = content_recs[i]
                score = content_weight
                if product['id'] not in seen_products:
                    hybrid_recommendations.append((product, score))
                    seen_products.add(product['id'])
                    
            if i < len(collab_recs):
                product = collab_recs[i]
                score = collab_weight
                if product['id'] not in seen_products:
                    hybrid_recommendations.append((product, score))
                    seen_products.add(product['id'])
        
        # Sort by score and return products
        hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
        return [prod for prod, _ in hybrid_recommendations[:n_recommendations]]

    def record_interaction(self, user_id, product_id, interaction_type, rating=None):
        """Record user interaction and update models"""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {
                "viewed": [],
                "purchased": [],
                "cart": [],
                "ratings": {}
            }
            
        user_data = self.user_interactions[user_id]
        
        if interaction_type == 'view':
            if product_id not in user_data['viewed']:
                user_data['viewed'].append(product_id)
        elif interaction_type == 'purchase':
            if product_id not in user_data['purchased']:
                user_data['purchased'].append(product_id)
        elif interaction_type == 'cart':
            if product_id not in user_data['cart']:
                user_data['cart'].append(product_id)
        elif interaction_type == 'rating' and rating is not None:
            user_data['ratings'][product_id] = rating
            
        # Update matrices and models
        self._initialize_matrices()

# Initialize products and recommendation engine
def initialize_user_interactions():
    """Initialize user interaction data"""
    try:
        # Try to load existing user interactions from a file
        with open('dataset/user_interactions.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, return default structure
        return {
            "user1": {
                "viewed": [],
                "purchased": [],
                "cart": [],
                "ratings": {}
            }
        }

def save_user_interactions():
    """Save user interactions to file"""
    try:
        with open('dataset/user_interactions.json', 'w') as f:
            json.dump(user_interactions, f)
    except Exception as e:
        logger.error(f"Failed to save user interactions: {e}")

def load_products():
    df = pd.read_csv('dataset/products.csv')
    products_dict = {}
    for _, row in df.iterrows():
        products_dict[row['id']] = {
            "id": row['id'],
            "name": row['name'],
            "category": row['category'],
            "price": row['price'],
            "description": row['description'],
            "tags": row['tags'],
            "rating": row['rating'],
            "views": row['views'],
            "purchases": row['purchases']
        }
    return products_dict
products = load_products()  # Assuming this function exists

user_interactions = initialize_user_interactions()
recommendation_engine = AdvancedRecommendationEngine(products, user_interactions)

import random
@app.route('/')
def index():
    random_ids = random.sample(list(products.keys()), 3)
    filtered_products = {pid: products[pid] for pid in random_ids}    
    return render_template('index.html', products=filtered_products)

@app.route('/api/recommendations/content/<int:product_id>')
def content_recommendations(product_id):
    recommendations = recommendation_engine.get_enhanced_content_recommendations(product_id)
    return jsonify(recommendations)

@app.route('/api/recommendations/collaborative/<user_id>')
def collaborative_recommendations(user_id):
    recommendations = recommendation_engine.get_enhanced_collaborative_recommendations(user_id)
    return jsonify(recommendations)

@app.route('/api/recommendations/hybrid/<user_id>/<int:product_id>')
def hybrid_recommendations(user_id, product_id):
    recommendations = recommendation_engine.get_enhanced_hybrid_recommendations(
        user_id, 
        product_id
    )
    return jsonify(recommendations)

@app.route('/api/interaction', methods=['POST'])
def record_interaction():
    data = request.json
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    interaction_type = data.get('type')
    rating = data.get('rating')
    
    if not all([user_id, product_id, interaction_type]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    recommendation_engine.record_interaction(user_id, product_id, interaction_type, rating)
    save_user_interactions()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)