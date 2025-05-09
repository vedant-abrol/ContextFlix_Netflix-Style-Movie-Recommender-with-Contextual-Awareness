import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

class ContentBasedNN(nn.Module):
    def __init__(self, n_genres, embedding_dim=128, hidden_dim=64):
        """
        Neural network for content-based filtering.
        
        Args:
            n_genres (int): Number of genre features
            embedding_dim (int): Size of the embedding layer
            hidden_dim (int): Size of the hidden layer
        """
        super(ContentBasedNN, self).__init__()
        
        # Network architecture
        self.genre_layer = nn.Sequential(
            nn.Linear(n_genres, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, genre_features):
        """
        Forward pass of the model.
        
        Args:
            genre_features (torch.Tensor): Batch of genre features
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        genre_embedding = self.genre_layer(genre_features)
        hidden = self.hidden_layer(genre_embedding)
        output = self.output_layer(hidden)
        
        # We add 3.0 to center predictions around the middle of the rating scale
        return output + 3.0


class ContentBasedModel:
    def __init__(self, data_path=None, embedding_dim=128, hidden_dim=64, 
                 learning_rate=0.001, batch_size=128, n_epochs=10, 
                 model_path=None, device=None):
        """
        Content-based movie recommendation model.
        
        Args:
            data_path (str): Path to the processed dataset
            embedding_dim (int): Size of the embedding layer
            hidden_dim (int): Size of the hidden layer
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            model_path (str): Path to save/load model weights
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_path = model_path or 'content_based_model.pt'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.genre_features = None
        self.genre_columns = None
        self.movie_indices = None
        self.movie_id_to_idx = None
        self.idx_to_movie_id = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load and preprocess data for content-based filtering.
        
        Args:
            data_path (str): Path to the processed dataset
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Extract genre columns (they start with 'Genre_')
        self.genre_columns = [col for col in self.data.columns if col.startswith('Genre_')]
        print(f"Found {len(self.genre_columns)} genre features: {self.genre_columns}")
        
        # Create movie features dataframe (unique movies with their genres)
        movie_features = self.data[['MovieID', 'Title'] + self.genre_columns].drop_duplicates(subset=['MovieID'])
        movie_features.set_index('MovieID', inplace=True)
        
        # Create mappings
        unique_movies = movie_features.index.unique()
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Extract genre features as a numpy array
        self.genre_features = movie_features[self.genre_columns].values
        self.movie_ids = movie_features.index.values
        self.movie_titles = movie_features['Title'].values
        
        print(f"Processed {len(unique_movies)} unique movies")
        
        # Initialize the model
        self.model = ContentBasedNN(
            n_genres=len(self.genre_columns),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
    
    def prepare_training_data(self, min_ratings=5):
        """
        Prepare training data for the content-based model.
        
        Args:
            min_ratings (int): Minimum number of ratings per user to include
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        # Group by user and filter users with enough ratings
        user_ratings_count = self.data.groupby('UserID').size()
        valid_users = user_ratings_count[user_ratings_count >= min_ratings].index
        
        train_data = self.data[self.data['UserID'].isin(valid_users)]
        print(f"Using {len(train_data)} ratings from {len(valid_users)} users who have at least {min_ratings} ratings")
        
        # Extract features and labels
        X = train_data[self.genre_columns].values
        y = train_data['Rating'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_val, y_train, y_val
    
    def train(self, save_model=True):
        """
        Train the content-based model.
        
        Args:
            save_model (bool): Whether to save the model after training
            
        Returns:
            tuple: (train_losses, val_losses)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        # Prepare training data
        X_train, X_val, y_train, y_val = self.prepare_training_data()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for genres, ratings in train_loader:
                # Forward pass
                outputs = self.model(genres)
                loss = criterion(outputs, ratings)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if save_model:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
        
        return train_losses, val_losses
    
    def load_model(self):
        """
        Load a trained model from disk.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def get_movie_embeddings(self):
        """
        Get embeddings for all movies.
        
        Returns:
            numpy.ndarray: Matrix of movie embeddings
        """
        if self.model is None or self.genre_features is None:
            raise ValueError("Model not initialized or trained.")
        
        genre_tensor = torch.FloatTensor(self.genre_features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get embeddings from the first hidden layer
            embeddings = self.model.genre_layer(genre_tensor).cpu().numpy()
        
        return embeddings
    
    def compute_similarity_matrix(self):
        """
        Compute similarity matrix between movies.
        
        Returns:
            numpy.ndarray: Similarity matrix
        """
        movie_embeddings = self.get_movie_embeddings()
        similarity_matrix = cosine_similarity(movie_embeddings)
        return similarity_matrix
    
    def recommend_similar_movies(self, movie_id, n=10):
        """
        Recommend similar movies based on content.
        
        Args:
            movie_id (int): Movie ID
            n (int): Number of recommendations
            
        Returns:
            pandas.DataFrame: DataFrame with similar movie recommendations
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix()
        
        # Get movie index
        try:
            movie_idx = self.movie_id_to_idx[movie_id]
        except KeyError:
            raise ValueError(f"Movie ID {movie_id} not found in training data.")
        
        # Get similarity scores
        movie_similarities = similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(movie_similarities)[::-1][1:n+1]
        similar_movie_ids = [self.idx_to_movie_id[idx] for idx in similar_indices]
        similarity_scores = movie_similarities[similar_indices]
        
        # Create recommendations dataframe
        movies_data = pd.DataFrame({
            'MovieID': self.movie_ids,
            'Title': self.movie_titles
        })
        
        recommendations = pd.DataFrame({
            'MovieID': similar_movie_ids,
            'SimilarityScore': similarity_scores
        })
        
        recommendations = pd.merge(recommendations, movies_data, on='MovieID', how='left')
        
        return recommendations[['MovieID', 'Title', 'SimilarityScore']]
    
    def predict_user_preferences(self, user_id, n=10):
        """
        Predict user's preferences for movies based on their genre preferences.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            
        Returns:
            pandas.DataFrame: DataFrame with recommended movies
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        # Get user's ratings
        user_ratings = self.data[self.data['UserID'] == user_id]
        
        if len(user_ratings) == 0:
            raise ValueError(f"User ID {user_id} not found in training data.")
        
        # Calculate user's genre preferences (weighted average of genre features by rating)
        user_genres = np.zeros(len(self.genre_columns))
        
        for _, row in user_ratings.iterrows():
            movie_id = row['MovieID']
            rating = row['Rating']
            rating_weight = rating / 5.0  # Normalize rating to [0, 1]
            
            # Get movie index
            if movie_id in self.movie_id_to_idx:
                movie_idx = self.movie_id_to_idx[movie_id]
                movie_genres = self.genre_features[movie_idx]
                user_genres += rating_weight * movie_genres
        
        # Normalize user genre preferences
        if np.sum(user_genres) > 0:
            user_genres = user_genres / np.sum(user_genres)
        
        # Predict ratings for all movies
        self.model.eval()
        genre_tensor = torch.FloatTensor(self.genre_features).to(self.device)
        
        with torch.no_grad():
            predicted_ratings = self.model(genre_tensor).cpu().numpy().flatten()
        
        # Get movies the user has already rated
        rated_movies = set(user_ratings['MovieID'].values)
        
        # Create a dataframe with predictions
        predictions = pd.DataFrame({
            'MovieID': self.movie_ids,
            'PredictedRating': predicted_ratings
        })
        
        # Exclude rated movies
        recommendations = predictions[~predictions['MovieID'].isin(rated_movies)]
        
        # Get top N recommendations
        recommendations = recommendations.sort_values('PredictedRating', ascending=False).head(n)
        
        # Get movie titles
        movies_data = pd.DataFrame({
            'MovieID': self.movie_ids,
            'Title': self.movie_titles
        })
        
        recommendations = pd.merge(recommendations, movies_data, on='MovieID', how='left')
        
        return recommendations[['MovieID', 'Title', 'PredictedRating']]
    
    def get_genre_importance(self, movie_id):
        """
        Get genre importance for a movie.
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            dict: Dictionary with genre importance scores
        """
        try:
            movie_idx = self.movie_id_to_idx[movie_id]
        except KeyError:
            raise ValueError(f"Movie ID {movie_id} not found in training data.")
        
        movie_genres = self.genre_features[movie_idx]
        
        # Get non-zero genre indices
        genres_present = np.where(movie_genres > 0)[0]
        
        if len(genres_present) == 0:
            return {}
        
        # Create importance dictionary
        genre_importance = {}
        for idx in genres_present:
            genre_name = self.genre_columns[idx].replace('Genre_', '')
            genre_importance[genre_name] = 1.0 / len(genres_present)
        
        return genre_importance


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/processed_movielens_data.csv"
    model = ContentBasedModel(
        data_path=data_path,
        embedding_dim=128,
        hidden_dim=64,
        learning_rate=0.001,
        batch_size=128,
        n_epochs=5,
        model_path="../models/content_based_model.pt"
    )
    
    # Train model
    train_losses, val_losses = model.train()
    
    # Example similar movie recommendation
    movie_id = 1  # Toy Story
    similar_movies = model.recommend_similar_movies(movie_id, n=5)
    print(f"Movies similar to MovieID {movie_id}:")
    print(similar_movies)
    
    # Example user preference prediction
    user_id = 1
    recommendations = model.predict_user_preferences(user_id, n=5)
    print(f"Recommendations for user {user_id} based on genre preferences:")
    print(recommendations)
