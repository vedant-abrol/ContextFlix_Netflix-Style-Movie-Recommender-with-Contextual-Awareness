from flask import Flask, request, jsonify
import os
import sys
import json
import pandas as pd
import numpy as np
import time

# Add the parent directory to sys.path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from models.hybrid import HybridRecommender

app = Flask(__name__)

# Initialize the recommender model
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       'data/processed/processed_movielens_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       'models')

# Setup models paths
MODEL_PATHS = {
    'collaborative': os.path.join(MODEL_DIR, 'collaborative_model.pt'),
    'content_based': os.path.join(MODEL_DIR, 'content_based_model.pt'),
    'sequential': os.path.join(MODEL_DIR, 'sequential_model.pt')
}
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pt')

# Initialize the recommender when the app starts
recommender = None

# Dictionary to store cached recommendations for faster response time
recommendation_cache = {}
cache_timeout = 3600  # Cache expiry in seconds (1 hour)

# Instead of @app.before_first_request, use an initialization function
def initialize_model():
    """Initialize the recommender model."""
    global recommender
    try:
        # Check if data file exists
        if not os.path.exists(DATA_PATH):
            print(f"Error: Data file not found at {DATA_PATH}")
            return jsonify({"error": "Model data not found. Please run data preprocessing first."}), 500

        # Initialize recommender
        print("Initializing recommender model...")
        recommender = HybridRecommender(
            data_path=DATA_PATH,
            model_paths=MODEL_PATHS,
            use_context=True,
            hybrid_model_path=HYBRID_MODEL_PATH
        )
        
        # Load pre-trained models
        print("Loading models...")
        recommender.load_models()
        
        print("Recommender system initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")

# Call initialization function at startup
initialize_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if recommender is not None:
        return jsonify({"status": "healthy", "model_loaded": True})
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get a list of available users."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get unique users
        users = recommender.data['UserID'].unique().tolist()
        
        # Add some user metadata if available
        user_info = []
        for user_id in users[:100]:  # Limit to first 100 users for performance
            user_data = recommender.data[recommender.data['UserID'] == user_id]
            rating_count = len(user_data)
            avg_rating = user_data['Rating'].mean()
            
            # Get user demographics if available
            demo_cols = ['Gender', 'Age', 'Occupation']
            demographics = {}
            for col in demo_cols:
                if col in user_data.columns:
                    demographics[col.lower()] = user_data[col].iloc[0]
            
            user_info.append({
                "id": int(user_id),
                "rating_count": rating_count,
                "avg_rating": float(avg_rating),
                **demographics
            })
        
        return jsonify({"users": user_info})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get a list of movies, with optional filters."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get query parameters
        genre = request.args.get('genre')
        search = request.args.get('search')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get unique movies
        movie_data = recommender.data[['MovieID', 'Title', 'Genres']].drop_duplicates()
        
        # Apply filters if specified
        if genre:
            movie_data = movie_data[movie_data['Genres'].str.contains(genre, case=False)]
        
        if search:
            movie_data = movie_data[movie_data['Title'].str.contains(search, case=False)]
        
        # Get total count before pagination
        total_count = len(movie_data)
        
        # Apply pagination
        movie_data = movie_data.iloc[offset:offset+limit]
        
        # Convert to list of dictionaries
        movies = movie_data.to_dict(orient='records')
        
        # Add placeholders for movie posters
        for movie in movies:
            # Extract year from title if present (e.g., "Toy Story (1995)" -> "1995")
            title = movie['Title']
            year = ""
            if "(" in title and ")" in title:
                start = title.rfind("(") + 1
                end = title.rfind(")")
                if start < end:
                    year = title[start:end]
            
            # Generate a placeholder with movie ID (could be replaced with actual poster URLs)
            movie['poster'] = f"https://via.placeholder.com/150x225.png?text={movie['MovieID']}"
            movie['year'] = year
        
        return jsonify({
            "movies": movies,
            "total": total_count,
            "offset": offset,
            "limit": limit
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend', methods=['GET'])
def recommend():
    """Get movie recommendations for a user."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get query parameters
        user_id = request.args.get('user_id', type=int)
        n = request.args.get('n', 10, type=int)
        time_of_day = request.args.get('time_of_day')
        device_type = request.args.get('device_type')
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if user_id is None:
            return jsonify({"error": "user_id parameter is required"}), 400
        
        # Set default time_of_day based on current time if not specified
        if not time_of_day:
            current_hour = time.localtime().tm_hour
            if 5 <= current_hour < 12:
                time_of_day = 'morning'
            elif 12 <= current_hour < 17:
                time_of_day = 'afternoon'
            elif 17 <= current_hour < 21:
                time_of_day = 'evening'
            else:
                time_of_day = 'night'
        
        # Set default device_type based on User-Agent if not specified
        if not device_type:
            user_agent = request.headers.get('User-Agent', '').lower()
            if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
                device_type = 'mobile'
            elif 'tablet' in user_agent or 'ipad' in user_agent:
                device_type = 'tablet'
            else:
                device_type = 'desktop'
        
        # Check cache if not refreshing
        cache_key = f"{user_id}_{n}_{time_of_day}_{device_type}"
        if not refresh and cache_key in recommendation_cache:
            cached_data, timestamp = recommendation_cache[cache_key]
            if time.time() - timestamp < cache_timeout:
                return jsonify(cached_data)
        
        # Generate recommendations
        recs = recommender.recommend_movies(
            user_id=user_id,
            n=n,
            time_of_day=time_of_day,
            device_type=device_type
        )
        
        # Process recommendations
        recommendations = []
        for _, row in recs.iterrows():
            movie_id = int(row['MovieID'])
            title = row['Title']
            
            # Extract year from title if present
            year = ""
            if "(" in title and ")" in title:
                start = title.rfind("(") + 1
                end = title.rfind(")")
                if start < end:
                    year = title[start:end]
            
            # Get explanation
            explanation = recommender.explain_recommendation(
                user_id, movie_id, time_of_day, device_type
            )
            
            # Get model contributions
            contributions = {
                "collaborative": float(row.get('CF_Contribution', 0.33)),
                "content_based": float(row.get('CB_Contribution', 0.33)),
                "sequential": float(row.get('Seq_Contribution', 0.34))
            }
            
            # Create recommendation item
            rec_item = {
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "score": float(row['HybridScore']),
                "poster": f"https://via.placeholder.com/150x225.png?text={movie_id}",
                "explanation": explanation,
                "contributions": contributions,
                "time_of_day": time_of_day,
                "device_type": device_type
            }
            
            # Add genres if available
            movie_data = recommender.data[recommender.data['MovieID'] == movie_id]
            if len(movie_data) > 0 and 'Genres' in movie_data.columns:
                genres = movie_data['Genres'].iloc[0].split('|')
                rec_item['genres'] = genres
            
            recommendations.append(rec_item)
        
        # Create response
        response = {
            "user_id": user_id,
            "time_of_day": time_of_day,
            "device_type": device_type,
            "count": len(recommendations),
            "recommendations": recommendations
        }
        
        # Cache the response
        recommendation_cache[cache_key] = (response, time.time())
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user_history', methods=['GET'])
def user_history():
    """Get a user's viewing history."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get query parameters
        user_id = request.args.get('user_id', type=int)
        limit = request.args.get('limit', 20, type=int)
        
        if user_id is None:
            return jsonify({"error": "user_id parameter is required"}), 400
        
        # Get user's ratings
        user_data = recommender.data[recommender.data['UserID'] == user_id]
        
        if len(user_data) == 0:
            return jsonify({"error": f"User ID {user_id} not found"}), 404
        
        # Sort by timestamp (newest first) and limit results
        user_data = user_data.sort_values('Timestamp', ascending=False).head(limit)
        
        # Process history items
        history = []
        for _, row in user_data.iterrows():
            movie_id = int(row['MovieID'])
            title = row['Title']
            rating = float(row['Rating'])
            timestamp = row['Timestamp']
            
            # Extract year from title if present
            year = ""
            if "(" in title and ")" in title:
                start = title.rfind("(") + 1
                end = title.rfind(")")
                if start < end:
                    year = title[start:end]
            
            # Create history item
            history_item = {
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "rating": rating,
                "timestamp": timestamp if isinstance(timestamp, str) else str(timestamp),
                "poster": f"https://via.placeholder.com/150x225.png?text={movie_id}"
            }
            
            # Add contextual features if available
            for ctx in ['Time_of_Day', 'Device_Type']:
                if ctx in row:
                    history_item[ctx.lower()] = row[ctx]
            
            # Add genres if available
            if 'Genres' in row:
                genres = row['Genres'].split('|')
                history_item['genres'] = genres
            
            history.append(history_item)
        
        # Create response
        response = {
            "user_id": user_id,
            "count": len(history),
            "history": history
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/similar_movies', methods=['GET'])
def similar_movies():
    """Get movies similar to a given movie."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get query parameters
        movie_id = request.args.get('movie_id', type=int)
        n = request.args.get('n', 10, type=int)
        
        if movie_id is None:
            return jsonify({"error": "movie_id parameter is required"}), 400
        
        # Get similar movies based on content
        similar = recommender.cb_model.recommend_similar_movies(movie_id, n=n)
        
        # Process similar movies
        movies = []
        for _, row in similar.iterrows():
            similar_id = int(row['MovieID'])
            title = row['Title']
            similarity = float(row['SimilarityScore'])
            
            # Extract year from title if present
            year = ""
            if "(" in title and ")" in title:
                start = title.rfind("(") + 1
                end = title.rfind(")")
                if start < end:
                    year = title[start:end]
            
            # Create movie item
            movie_item = {
                "movie_id": similar_id,
                "title": title,
                "year": year,
                "similarity": similarity,
                "poster": f"https://via.placeholder.com/150x225.png?text={similar_id}"
            }
            
            # Add genres if available
            movie_data = recommender.data[recommender.data['MovieID'] == similar_id]
            if len(movie_data) > 0 and 'Genres' in movie_data.columns:
                genres = movie_data['Genres'].iloc[0].split('|')
                movie_item['genres'] = genres
            
            movies.append(movie_item)
        
        # Get the original movie details
        original_movie = recommender.data[recommender.data['MovieID'] == movie_id].iloc[0]
        original_title = original_movie['Title']
        
        # Create response
        response = {
            "movie_id": movie_id,
            "title": original_title,
            "count": len(movies),
            "similar_movies": movies
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/movie_genres', methods=['GET'])
def movie_genres():
    """Get all available movie genres."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Extract all genres from the dataset
        all_genres = set()
        for genres in recommender.data['Genres'].unique():
            for genre in genres.split('|'):
                all_genres.add(genre)
        
        # Get genre counts
        genre_counts = {}
        for genre in all_genres:
            count = recommender.data[recommender.data['Genres'].str.contains(genre)]['MovieID'].nunique()
            genre_counts[genre] = count
        
        # Sort by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create response
        response = {
            "genres": [{"name": genre, "count": count} for genre, count in sorted_genres]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user_preferences', methods=['GET'])
def user_preferences():
    """Get a user's genre preferences based on their ratings."""
    if recommender is None:
        return jsonify({"error": "Model not initialized"}), 503
    
    try:
        # Get query parameters
        user_id = request.args.get('user_id', type=int)
        
        if user_id is None:
            return jsonify({"error": "user_id parameter is required"}), 400
        
        # Get user's ratings
        user_data = recommender.data[recommender.data['UserID'] == user_id]
        
        if len(user_data) == 0:
            return jsonify({"error": f"User ID {user_id} not found"}), 404
        
        # Extract genre columns
        genre_columns = [col for col in user_data.columns if col.startswith('Genre_')]
        
        if not genre_columns:
            return jsonify({"error": "Genre features not available in the dataset"}), 500
        
        # Calculate weighted average of genre features based on ratings
        user_genres = {}
        
        for _, row in user_data.iterrows():
            rating_weight = row['Rating'] / 5.0  # Normalize rating to [0, 1]
            
            for genre_col in genre_columns:
                genre = genre_col.replace('Genre_', '')
                if row[genre_col] > 0:  # Movie has this genre
                    if genre not in user_genres:
                        user_genres[genre] = {'weight': 0, 'count': 0}
                    
                    user_genres[genre]['weight'] += rating_weight
                    user_genres[genre]['count'] += 1
        
        # Normalize weights
        for genre in user_genres:
            if user_genres[genre]['count'] > 0:
                user_genres[genre]['average_rating'] = user_genres[genre]['weight'] / user_genres[genre]['count'] * 5.0
            else:
                user_genres[genre]['average_rating'] = 0
        
        # Sort by weight
        sorted_preferences = sorted(
            [(k, v['count'], v['average_rating']) for k, v in user_genres.items()],
            key=lambda x: (x[1], x[2]),  # Sort by count, then by average rating
            reverse=True
        )
        
        # Create response
        preferences = [
            {
                "genre": genre.replace('_', ' '),
                "count": count,
                "average_rating": float(avg_rating)
            }
            for genre, count, avg_rating in sorted_preferences
        ]
        
        response = {
            "user_id": user_id,
            "preference_count": len(preferences),
            "preferences": preferences
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Cross-Origin Resource Sharing (CORS) support
@app.after_request
def after_request(response):
    """Add CORS headers to allow cross-origin requests."""
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    header['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 