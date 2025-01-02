# E-commerce Recommendation System

A sophisticated recommendation engine that combines collaborative filtering, content-based filtering, and hybrid approaches to provide personalized product recommendations.

## Features

- **Content-based Filtering**: Recommends products based on item features and descriptions
- **Collaborative Filtering**: Suggests items based on user interaction patterns
- **Hybrid Recommendations**: Combines both approaches for better accuracy
- **User Interactions**:
  - Product views
  - Add to cart
  - Purchases
  - Ratings (1-5 stars)
- **Real-time Updates**: Recommendations update instantly after user interactions
- **CSRF Protection**: Secure API endpoints

## Tech Stack

- **Backend**: Python 3.12, Flask
- **Frontend**: HTML, CSS, JavaScript, jQuery
- **Machine Learning**: scikit-learn, implicit
- **Data Processing**: pandas, numpy, scipy
- **Database**: JSON file storage (user interactions)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce-recommendations.git
cd ecommerce-recommendations

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

recommendation-system/
├── [app.py]          # Main Flask application
├── static/
│   ├── style.css         # CSS styles
│   └── script.js         # Frontend JavaScript
├── templates/
│   └── index.html        # Main HTML template
├── dataset/
│   ├── products.csv      # Product catalog
│   └── user_interactions.json  # User interaction data
└── [requirements.txt]  # Python dependencies