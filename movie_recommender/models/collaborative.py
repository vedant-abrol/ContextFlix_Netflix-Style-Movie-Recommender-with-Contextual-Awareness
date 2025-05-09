import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        """
        Matrix Factorization model for collaborative filtering.
        
        Args:
            n_users (int): Number of users
            n_movies (int): Number of movies
            n_factors (int): Size of latent factors
        """
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.movie_embeddings = nn.Embedding(n_movies, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.movie_biases = nn.Embedding(n_movies, 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.movie_embeddings.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)
        
    def forward(self, user_indices, movie_indices):
        """
        Forward pass of the model.
        
        Args:
            user_indices (torch.Tensor): Batch of user indices
            movie_indices (torch.Tensor): Batch of movie indices
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        user_embedding = self.user_embeddings(user_indices)
        movie_embedding = self.movie_embeddings(movie_indices)
        user_bias = self.user_biases(user_indices).squeeze()
        movie_bias = self.movie_biases(movie_indices).squeeze()
        
        # Compute dot product of user and movie embeddings
        dot_product = torch.sum(user_embedding * movie_embedding, dim=1)
        
        # Add biases and global bias (we use 3.5 as global bias which is middle of 1-5 scale)
        prediction = dot_product + user_bias + movie_bias + 3.5
        
        return prediction


class CollaborativeFilteringModel:
    def __init__(self, data_path=None, n_factors=50, learning_rate=0.001, weight_decay=1e-5, 
                 batch_size=1024, n_epochs=20, model_path=None, device=None):
        """
        Collaborative Filtering model trainer and inferencer.
        
        Args:
            data_path (str): Path to processed dataset
            n_factors (int): Size of latent factors
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): L2 regularization parameter
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            model_path (str): Path to save/load model weights
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_path = model_path or 'collaborative_model.pt'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.user_mapping = None
        self.movie_mapping = None
        self.inverse_user_mapping = None
        self.inverse_movie_mapping = None
        
        if data_path:
            self.load_data(data_path)
        
    def load_data(self, data_path):
        """
        Load and preprocess data for collaborative filtering.
        
        Args:
            data_path (str): Path to processed dataset
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Create mappings for user and movie IDs
        unique_users = self.data['UserID'].unique()
        unique_movies = self.data['MovieID'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_mapping = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        
        # Create inverse mappings for prediction
        self.inverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.inverse_movie_mapping = {idx: movie_id for movie_id, idx in self.movie_mapping.items()}
        
        # Map IDs to indices
        self.data['UserIdx'] = self.data['UserID'].map(self.user_mapping)
        self.data['MovieIdx'] = self.data['MovieID'].map(self.movie_mapping)
        
        print(f"Processed data: {len(unique_users)} users, {len(unique_movies)} movies, {len(self.data)} ratings")
        
        # Create the model
        self.model = MatrixFactorization(
            n_users=len(unique_users),
            n_movies=len(unique_movies),
            n_factors=self.n_factors
        ).to(self.device)
    
    def prepare_dataloaders(self, test_size=0.2, random_state=42):
        """
        Prepare DataLoader objects for training and validation.
        
        Args:
            test_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        train_data, val_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state
        )
        
        # Convert to tensors
        train_users = torch.LongTensor(train_data['UserIdx'].values)
        train_movies = torch.LongTensor(train_data['MovieIdx'].values)
        train_ratings = torch.FloatTensor(train_data['Rating'].values)
        
        val_users = torch.LongTensor(val_data['UserIdx'].values)
        val_movies = torch.LongTensor(val_data['MovieIdx'].values)
        val_ratings = torch.FloatTensor(val_data['Rating'].values)
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(train_users, train_movies, train_ratings)
        val_dataset = torch.utils.data.TensorDataset(val_users, val_movies, val_ratings)
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, save_model=True):
        """
        Train the collaborative filtering model.
        
        Args:
            save_model (bool): Whether to save the model after training
            
        Returns:
            tuple: (train_losses, val_losses, val_rmse)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_dataloaders()
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_rmse = []
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (users, movies, ratings) in enumerate(train_loader):
                users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                
                # Forward pass
                outputs = self.model(users, movies)
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
            val_loss = 0.0
            all_preds = []
            all_true = []
            
            with torch.no_grad():
                for users, movies, ratings in val_loader:
                    users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                    outputs = self.model(users, movies)
                    loss = criterion(outputs, ratings)
                    val_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_true.extend(ratings.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(all_true, all_preds))
            val_rmse.append(rmse)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {rmse:.4f}")
        
        if save_model:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
        
        return train_losses, val_losses, val_rmse
    
    def load_model(self):
        """
        Load a trained model from disk.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a specific user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        # Convert IDs to indices
        user_idx = self.user_mapping.get(user_id)
        movie_idx = self.movie_mapping.get(movie_id)
        
        if user_idx is None or movie_idx is None:
            raise ValueError(f"User ID {user_id} or Movie ID {movie_id} not found in training data.")
        
        # Convert to tensors
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, movie_tensor).item()
        
        # Clip to rating range [1, 5]
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction
    
    def get_user_embeddings(self, user_id):
        """
        Get the embedding vector for a specific user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            numpy.ndarray: User embedding vector
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in training data.")
        
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        with torch.no_grad():
            user_embedding = self.model.user_embeddings(user_tensor).cpu().numpy()[0]
        
        return user_embedding
    
    def get_movie_embeddings(self, movie_id):
        """
        Get the embedding vector for a specific movie.
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            numpy.ndarray: Movie embedding vector
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        movie_idx = self.movie_mapping.get(movie_id)
        if movie_idx is None:
            raise ValueError(f"Movie ID {movie_id} not found in training data.")
        
        movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
        
        with torch.no_grad():
            movie_embedding = self.model.movie_embeddings(movie_tensor).cpu().numpy()[0]
        
        return movie_embedding
    
    def recommend_movies(self, user_id, n=10, exclude_rated=True):
        """
        Recommend top N movies for a user.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations to generate
            exclude_rated (bool): Whether to exclude movies the user has already rated
            
        Returns:
            pandas.DataFrame: DataFrame with movie recommendations and predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in training data.")
        
        # Get user tensor
        user_tensor = torch.LongTensor([user_idx] * len(self.movie_mapping)).to(self.device)
        
        # Get movie indices
        movie_indices = list(self.movie_mapping.values())
        movie_tensor = torch.LongTensor(movie_indices).to(self.device)
        
        # Get movies the user has already rated
        if exclude_rated:
            rated_movies = set(self.data[self.data['UserID'] == user_id]['MovieID'].values)
        else:
            rated_movies = set()
        
        # Predict ratings for all movies
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor).cpu().numpy()
        
        # Create a dataframe with predictions
        movie_ids = [self.inverse_movie_mapping[idx] for idx in movie_indices]
        pred_df = pd.DataFrame({
            'MovieID': movie_ids,
            'PredictedRating': predictions
        })
        
        # Exclude rated movies if requested
        if exclude_rated:
            pred_df = pred_df[~pred_df['MovieID'].isin(rated_movies)]
        
        # Get top N recommendations
        recommendations = pred_df.sort_values('PredictedRating', ascending=False).head(n)
        
        # Merge with movie data to get movie titles
        movie_data = self.data[['MovieID', 'Title']].drop_duplicates()
        recommendations = pd.merge(recommendations, movie_data, on='MovieID', how='left')
        
        return recommendations[['MovieID', 'Title', 'PredictedRating']]


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/processed_movielens_data.csv"
    model = CollaborativeFilteringModel(
        data_path=data_path,
        n_factors=50,
        learning_rate=0.001,
        weight_decay=1e-5,
        batch_size=1024,
        n_epochs=5,
        model_path="../models/collaborative_model.pt"
    )
    
    # Train model
    train_losses, val_losses, val_rmse = model.train()
    
    # Example recommendation
    user_id = 1
    recommendations = model.recommend_movies(user_id, n=10)
    print(f"Top 10 recommendations for user {user_id}:")
    print(recommendations)
