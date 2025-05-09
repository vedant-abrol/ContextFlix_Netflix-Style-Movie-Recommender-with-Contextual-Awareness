import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import MinMaxScaler
import json

# Import our model components
from .collaborative import CollaborativeFilteringModel
from .content_based import ContentBasedModel
from .sequential import SequentialModel

class HybridCombiner(nn.Module):
    def __init__(self, n_models=3, hidden_dim=16):
        """
        Neural network for combining predictions from multiple models.
        
        Args:
            n_models (int): Number of base models
            hidden_dim (int): Size of hidden layer
        """
        super(HybridCombiner, self).__init__()
        
        # Model weights combiner
        self.combiner = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Scale to [0, 1]
        )
        
        # Initialize weights
        for layer in self.combiner:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, model_scores):
        """
        Forward pass of the combiner model.
        
        Args:
            model_scores (torch.Tensor): Scores from different models [batch_size, n_models]
            
        Returns:
            torch.Tensor: Combined scores [batch_size, 1]
        """
        return self.combiner(model_scores)


class ContextualWeightNet(nn.Module):
    def __init__(self, n_time_contexts=4, n_device_contexts=4, n_models=3, hidden_dim=16):
        """
        Neural network that adjusts model weights based on context.
        
        Args:
            n_time_contexts (int): Number of time contexts (e.g., morning, afternoon, evening, night)
            n_device_contexts (int): Number of device contexts (e.g., mobile, tablet, desktop, TV)
            n_models (int): Number of base models
            hidden_dim (int): Size of hidden layer
        """
        super(ContextualWeightNet, self).__init__()
        
        # Context embeddings
        self.time_embeddings = nn.Embedding(n_time_contexts, hidden_dim)
        self.device_embeddings = nn.Embedding(n_device_contexts, hidden_dim)
        
        # Combine context embeddings
        self.context_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Output layer that produces weights for each model
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, n_models),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
        
        # Initialize weights
        nn.init.normal_(self.time_embeddings.weight, mean=0, std=0.01)
        nn.init.normal_(self.device_embeddings.weight, mean=0, std=0.01)
        for layer in self.context_combiner:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.weight_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, time_indices, device_indices):
        """
        Forward pass to generate context-dependent weights.
        
        Args:
            time_indices (torch.Tensor): Time context indices [batch_size]
            device_indices (torch.Tensor): Device context indices [batch_size]
            
        Returns:
            torch.Tensor: Model weights [batch_size, n_models]
        """
        time_emb = self.time_embeddings(time_indices)
        device_emb = self.device_embeddings(device_indices)
        
        # Concatenate embeddings
        context_features = torch.cat([time_emb, device_emb], dim=1)
        
        # Combine context features
        combined = self.context_combiner(context_features)
        
        # Predict weights
        weights = self.weight_predictor(combined)
        
        return weights


class HybridRecommender:
    def __init__(self, data_path=None, model_paths=None, use_context=True, learning_rate=0.001, 
                 batch_size=128, n_epochs=5, hybrid_model_path=None, device=None):
        """
        Hybrid recommender system combining collaborative, content-based, and sequential models.
        
        Args:
            data_path (str): Path to the processed dataset
            model_paths (dict): Paths to pretrained model weights
            use_context (bool): Whether to use contextual features for weighting
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            hybrid_model_path (str): Path to save/load hybrid model weights
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.data_path = data_path
        self.model_paths = model_paths or {
            'collaborative': 'models/collaborative_model.pt',
            'content_based': 'models/content_based_model.pt',
            'sequential': 'models/sequential_model.pt'
        }
        self.use_context = use_context
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hybrid_model_path = hybrid_model_path or 'models/hybrid_model.pt'
        self.context_model_path = 'models/context_model.pt'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize sub-models
        self.cf_model = None
        self.cb_model = None
        self.seq_model = None
        
        # Initialize hybrid combiner
        self.hybrid_model = HybridCombiner(n_models=3).to(self.device)
        
        # Initialize contextual weighting model (if used)
        if self.use_context:
            self.context_model = ContextualWeightNet(
                n_time_contexts=4,  # morning, afternoon, evening, night
                n_device_contexts=4,  # mobile, tablet, desktop, TV
                n_models=3
            ).to(self.device)
        else:
            self.context_model = None
        
        # Initialize scalers for normalizing scores
        self.score_scalers = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load data and initialize all component models.
        
        Args:
            data_path (str): Path to the processed dataset
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Map context to indices
        if self.use_context:
            time_mapping = {
                'morning': 0,
                'afternoon': 1,
                'evening': 2,
                'night': 3
            }
            device_mapping = {
                'mobile': 0,
                'tablet': 1,
                'desktop': 2,
                'TV': 3
            }
            self.time_mapping = time_mapping
            self.device_mapping = device_mapping
            
            # Define inverse mappings
            self.inv_time_mapping = {v: k for k, v in time_mapping.items()}
            self.inv_device_mapping = {v: k for k, v in device_mapping.items()}
        
        # Initialize base models
        print("Initializing collaborative filtering model...")
        self.cf_model = CollaborativeFilteringModel(
            data_path=data_path,
            model_path=self.model_paths.get('collaborative'),
            device=self.device
        )
        
        print("Initializing content-based model...")
        self.cb_model = ContentBasedModel(
            data_path=data_path,
            model_path=self.model_paths.get('content_based'),
            device=self.device
        )
        
        print("Initializing sequential model...")
        self.seq_model = SequentialModel(
            data_path=data_path,
            model_path=self.model_paths.get('sequential'),
            use_context=self.use_context,
            device=self.device
        )
        
        # Initialize score scalers
        self.score_scalers = {
            'collaborative': MinMaxScaler(),
            'content_based': MinMaxScaler(),
            'sequential': MinMaxScaler()
        }
    
    def prepare_training_data(self, test_ratio=0.2):
        """
        Prepare training data for the hybrid model.
        
        Args:
            test_ratio (float): Proportion of data to use for testing
            
        Returns:
            tuple: (train_data, test_data)
        """
        # Get a sample of user-movie pairs with ground truth ratings
        user_movie_pairs = self.data[['UserID', 'MovieID', 'Rating', 'Time_of_Day', 'Device_Type']].drop_duplicates()
        
        # Map time and device to indices
        if self.use_context:
            user_movie_pairs['TimeIdx'] = user_movie_pairs['Time_of_Day'].map(self.time_mapping)
            user_movie_pairs['DeviceIdx'] = user_movie_pairs['Device_Type'].map(self.device_mapping)
        
        # Split into train and test
        np.random.seed(42)
        msk = np.random.rand(len(user_movie_pairs)) < (1 - test_ratio)
        train_data = user_movie_pairs[msk]
        test_data = user_movie_pairs[~msk]
        
        print(f"Created {len(train_data)} training samples and {len(test_data)} test samples")
        
        return train_data, test_data
    
    def generate_model_predictions(self, data):
        """
        Generate predictions from all base models for a set of user-movie pairs.
        
        Args:
            data (pandas.DataFrame): DataFrame with UserID, MovieID, and context features
            
        Returns:
            pandas.DataFrame: DataFrame with predictions from all models
        """
        predictions = data.copy()
        
        # Initialize prediction columns
        predictions['CF_Score'] = 0.0
        predictions['CB_Score'] = 0.0
        predictions['Seq_Score'] = 0.0
        
        # Generate predictions from each model
        print("Generating collaborative filtering predictions...")
        for i, row in predictions.iterrows():
            try:
                # Collaborative filtering prediction
                predictions.at[i, 'CF_Score'] = self.cf_model.predict(row['UserID'], row['MovieID'])
                
                # Content-based prediction
                user_preferences = self.cb_model.predict_user_preferences(row['UserID'], n=1)
                if row['MovieID'] in user_preferences['MovieID'].values:
                    content_score = user_preferences[user_preferences['MovieID'] == row['MovieID']]['PredictedRating'].values[0]
                else:
                    # If we don't have a direct prediction, use movie similarity
                    # Get user's top rated movie
                    user_ratings = self.data[self.data['UserID'] == row['UserID']].sort_values('Rating', ascending=False)
                    if len(user_ratings) > 0:
                        top_movie = user_ratings.iloc[0]['MovieID']
                        similar_movies = self.cb_model.recommend_similar_movies(top_movie, n=100)
                        if row['MovieID'] in similar_movies['MovieID'].values:
                            similarity = similar_movies[similar_movies['MovieID'] == row['MovieID']]['SimilarityScore'].values[0]
                            content_score = 3.0 + similarity  # Base rating + similarity adjustment
                        else:
                            content_score = 3.0  # Default neutral rating
                    else:
                        content_score = 3.0
                
                predictions.at[i, 'CB_Score'] = content_score
                
                # Sequential prediction
                if self.use_context:
                    seq_recs = self.seq_model.recommend_next_movies(
                        row['UserID'], 
                        n=100, 
                        time_of_day=row['Time_of_Day'], 
                        device_type=row['Device_Type']
                    )
                else:
                    seq_recs = self.seq_model.recommend_next_movies(row['UserID'], n=100)
                
                if row['MovieID'] in seq_recs['MovieID'].values:
                    seq_score = seq_recs[seq_recs['MovieID'] == row['MovieID']]['Score'].values[0]
                    # Map sequential scores (logits) to a rating-like scale
                    seq_score = (seq_score + 10) / 20 * 4 + 1  # Map from (-10, 10) to (1, 5)
                else:
                    seq_score = 3.0
                
                predictions.at[i, 'Seq_Score'] = seq_score
                
            except Exception as e:
                print(f"Error generating predictions for user {row['UserID']}, movie {row['MovieID']}: {e}")
                # Keep default values
        
        # Scale scores to [0, 1] for better combining
        for model_name in ['CF', 'CB', 'Seq']:
            score_col = f'{model_name}_Score'
            self.score_scalers[model_name.lower()].fit(predictions[[score_col]])
            predictions[f'{model_name}_Score_Scaled'] = self.score_scalers[model_name.lower()].transform(predictions[[score_col]])
        
        return predictions
    
    def train(self, save_model=True):
        """
        Train the hybrid recommendation model.
        
        Args:
            save_model (bool): Whether to save the model after training
            
        Returns:
            tuple: (train_loss, test_rmse)
        """
        # Prepare data
        train_data, test_data = self.prepare_training_data()
        
        # Generate predictions from base models
        train_predictions = self.generate_model_predictions(train_data)
        
        # Convert to tensors for training
        input_scores = torch.FloatTensor(
            train_predictions[['CF_Score_Scaled', 'CB_Score_Scaled', 'Seq_Score_Scaled']].values
        ).to(self.device)
        
        target_ratings = torch.FloatTensor(
            train_predictions['Rating'].values / 5.0  # Scale to [0, 1]
        ).reshape(-1, 1).to(self.device)
        
        if self.use_context:
            time_indices = torch.LongTensor(train_predictions['TimeIdx'].values).to(self.device)
            device_indices = torch.LongTensor(train_predictions['DeviceIdx'].values).to(self.device)
        
        # Create DataLoader
        if self.use_context:
            dataset = torch.utils.data.TensorDataset(input_scores, time_indices, device_indices, target_ratings)
        else:
            dataset = torch.utils.data.TensorDataset(input_scores, target_ratings)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        
        if self.use_context:
            optimizer = optim.Adam([
                {'params': self.hybrid_model.parameters()},
                {'params': self.context_model.parameters()}
            ], lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.hybrid_model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        
        for epoch in range(self.n_epochs):
            self.hybrid_model.train()
            if self.use_context:
                self.context_model.train()
            
            epoch_loss = 0.0
            
            for batch in data_loader:
                if self.use_context:
                    scores, time_idx, device_idx, ratings = batch
                    
                    # Get context-dependent weights
                    weights = self.context_model(time_idx, device_idx)
                    
                    # Apply weights to model scores
                    weighted_scores = scores * weights
                    
                    # Sum weighted scores
                    combined_scores = torch.sum(weighted_scores, dim=1, keepdim=True)
                else:
                    scores, ratings = batch
                    combined_scores = self.hybrid_model(scores)
                
                loss = criterion(combined_scores, ratings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(data_loader)
            train_losses.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{self.n_epochs} | Train Loss: {epoch_loss:.4f}")
        
        if save_model:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.hybrid_model_path), exist_ok=True)
            torch.save(self.hybrid_model.state_dict(), self.hybrid_model_path)
            print(f"Hybrid model saved to {self.hybrid_model_path}")
            
            if self.use_context:
                torch.save(self.context_model.state_dict(), self.context_model_path)
                print(f"Context model saved to {self.context_model_path}")
        
        # Evaluate on test data
        test_rmse = self.evaluate(test_data)
        print(f"Test RMSE: {test_rmse:.4f}")
        
        return train_losses, test_rmse
    
    def load_models(self):
        """
        Load all trained models.
        """
        try:
            # Load base models
            self.cf_model.load_model()
            self.cb_model.load_model()
            self.seq_model.load_model()
            
            # Load hybrid model
            self.hybrid_model.load_state_dict(torch.load(self.hybrid_model_path, map_location=self.device))
            self.hybrid_model.eval()
            print(f"Hybrid model loaded from {self.hybrid_model_path}")
            
            # Load context model if used
            if self.use_context and os.path.exists(self.context_model_path):
                self.context_model.load_state_dict(torch.load(self.context_model_path, map_location=self.device))
                self.context_model.eval()
                print(f"Context model loaded from {self.context_model_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def evaluate(self, test_data):
        """
        Evaluate the hybrid model on test data.
        
        Args:
            test_data (pandas.DataFrame): Test data with UserID, MovieID, Rating
            
        Returns:
            float: RMSE on test data
        """
        # Generate predictions from base models
        test_predictions = self.generate_model_predictions(test_data)
        
        # Convert to tensors
        input_scores = torch.FloatTensor(
            test_predictions[['CF_Score_Scaled', 'CB_Score_Scaled', 'Seq_Score_Scaled']].values
        ).to(self.device)
        
        if self.use_context:
            time_indices = torch.LongTensor(test_predictions['TimeIdx'].values).to(self.device)
            device_indices = torch.LongTensor(test_predictions['DeviceIdx'].values).to(self.device)
        
        # Get hybrid predictions
        self.hybrid_model.eval()
        if self.use_context:
            self.context_model.eval()
        
        with torch.no_grad():
            if self.use_context:
                # Get context-dependent weights
                weights = self.context_model(time_indices, device_indices)
                
                # Apply weights to model scores
                weighted_scores = input_scores * weights
                
                # Sum weighted scores
                combined_scores = torch.sum(weighted_scores, dim=1, keepdim=True)
            else:
                combined_scores = self.hybrid_model(input_scores)
            
            # Scale back to [1, 5]
            hybrid_ratings = combined_scores.cpu().numpy() * 5.0
            
        # Calculate RMSE
        true_ratings = test_predictions['Rating'].values.reshape(-1, 1)
        rmse = np.sqrt(np.mean((hybrid_ratings - true_ratings) ** 2))
        
        return rmse
    
    def recommend_movies(self, user_id, n=10, time_of_day=None, device_type=None, exclude_rated=True):
        """
        Generate movie recommendations for a user using the hybrid model.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            time_of_day (str, optional): Current time of day (morning, afternoon, evening, night)
            device_type (str, optional): Current device type (mobile, tablet, desktop, TV)
            exclude_rated (bool): Whether to exclude movies the user has already rated
            
        Returns:
            pandas.DataFrame: DataFrame with recommended movies
        """
        # Set default context if not provided
        if self.use_context:
            time_of_day = time_of_day or 'evening'
            device_type = device_type or 'TV'
            time_idx = self.time_mapping.get(time_of_day, 0)
            device_idx = self.device_mapping.get(device_type, 3)  # Default to TV
        
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend_movies(user_id, n=n*2, exclude_rated=exclude_rated)
        cb_recs = self.cb_model.predict_user_preferences(user_id, n=n*2)
        
        if self.use_context:
            seq_recs = self.seq_model.recommend_next_movies(
                user_id, n=n*2, time_of_day=time_of_day, device_type=device_type
            )
        else:
            seq_recs = self.seq_model.recommend_next_movies(user_id, n=n*2)
        
        # Combine all unique movie IDs
        all_movie_ids = set(cf_recs['MovieID']) | set(cb_recs['MovieID']) | set(seq_recs['MovieID'])
        
        # Create a DataFrame with all candidate movies
        candidates = pd.DataFrame({'MovieID': list(all_movie_ids)})
        
        # Merge with ratings data to get movie titles
        movie_data = self.data[['MovieID', 'Title']].drop_duplicates()
        candidates = pd.merge(candidates, movie_data, on='MovieID', how='left')
        
        # Get scores from each model for all candidates
        candidates = self.generate_model_predictions(
            pd.DataFrame({
                'UserID': [user_id] * len(candidates),
                'MovieID': candidates['MovieID'],
                'Rating': [0.0] * len(candidates),  # Placeholder
                'Time_of_Day': [time_of_day] * len(candidates) if self.use_context else [None] * len(candidates),
                'Device_Type': [device_type] * len(candidates) if self.use_context else [None] * len(candidates)
            })
        )
        
        # Convert model scores to tensors
        input_scores = torch.FloatTensor(
            candidates[['CF_Score_Scaled', 'CB_Score_Scaled', 'Seq_Score_Scaled']].values
        ).to(self.device)
        
        # Get hybrid predictions
        self.hybrid_model.eval()
        if self.use_context:
            self.context_model.eval()
            time_indices = torch.LongTensor([time_idx] * len(candidates)).to(self.device)
            device_indices = torch.LongTensor([device_idx] * len(candidates)).to(self.device)
        
        with torch.no_grad():
            if self.use_context:
                # Get context-dependent weights
                weights = self.context_model(time_indices, device_indices)
                
                # Get and store model weights for explanation
                model_weights = weights[0].cpu().numpy()
                
                # Apply weights to model scores
                weighted_scores = input_scores * weights
                
                # Sum weighted scores
                combined_scores = torch.sum(weighted_scores, dim=1, keepdim=True)
            else:
                combined_scores = self.hybrid_model(input_scores)
                model_weights = np.array([0.33, 0.33, 0.34])  # Default equal weights
            
            # Scale back to [1, 5]
            hybrid_ratings = combined_scores.cpu().numpy() * 5.0
        
        # Add hybrid scores to candidates
        candidates['HybridScore'] = hybrid_ratings
        
        # Get top N recommendations
        recommendations = candidates.sort_values('HybridScore', ascending=False).head(n)
        
        # Add model contributions for explainability
        for i, row in recommendations.iterrows():
            cf_contrib = row['CF_Score_Scaled'] * model_weights[0]
            cb_contrib = row['CB_Score_Scaled'] * model_weights[1]
            seq_contrib = row['Seq_Score_Scaled'] * model_weights[2]
            
            total_contrib = cf_contrib + cb_contrib + seq_contrib
            if total_contrib > 0:
                recommendations.at[i, 'CF_Contribution'] = cf_contrib / total_contrib
                recommendations.at[i, 'CB_Contribution'] = cb_contrib / total_contrib
                recommendations.at[i, 'Seq_Contribution'] = seq_contrib / total_contrib
            else:
                recommendations.at[i, 'CF_Contribution'] = 0.33
                recommendations.at[i, 'CB_Contribution'] = 0.33
                recommendations.at[i, 'Seq_Contribution'] = 0.34
        
        # Add context information if used
        if self.use_context:
            recommendations['Time_of_Day'] = time_of_day
            recommendations['Device_Type'] = device_type
        
        return recommendations
    
    def explain_recommendation(self, user_id, movie_id, time_of_day=None, device_type=None):
        """
        Generate an explanation for a recommendation.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            time_of_day (str, optional): Time of day
            device_type (str, optional): Device type
            
        Returns:
            dict: Explanation data
        """
        try:
            # Set default context if not provided
            if self.use_context:
                time_of_day = time_of_day or 'evening'
                device_type = device_type or 'TV'
            
            # Get movie title
            movie_data = self.data[self.data['MovieID'] == movie_id]
            if len(movie_data) == 0:
                return {"error": f"Movie ID {movie_id} not found."}
            
            movie_title = movie_data['Title'].iloc[0]
            
            # Get predictions from individual models
            # Collaborative filtering explanation
            try:
                similar_users = "users with similar taste" # Placeholder - we would need to implement this
                cf_predicted_rating = self.cf_model.predict(user_id, movie_id)
                cf_explanation = f"Similar users rated this movie {cf_predicted_rating:.1f}/5"
            except Exception as e:
                cf_explanation = "Could not generate collaborative filtering explanation."
            
            # Content-based explanation
            try:
                genre_importance = self.cb_model.get_genre_importance(movie_id)
                top_genres = sorted(genre_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                genres_text = ", ".join([g.replace("_", " ") for g, _ in top_genres])
                cb_explanation = f"Matches your preference for {genres_text}"
            except Exception as e:
                cb_explanation = "Could not generate content-based explanation."
            
            # Sequential explanation
            try:
                seq_explanation = self.seq_model.explain_sequential_recommendation(user_id, movie_id)
                seq_text = seq_explanation.get("explanation", "Based on your viewing history.")
            except Exception as e:
                seq_text = "Could not generate sequential explanation."
            
            # Prepare explanation
            explanation = {
                "movie": {
                    "id": movie_id,
                    "title": movie_title
                },
                "collaborative_filtering": {
                    "explanation": cf_explanation
                },
                "content_based": {
                    "explanation": cb_explanation,
                    "genres": [{"name": g.replace("_", " "), "importance": i} for g, i in top_genres] if 'top_genres' in locals() else []
                },
                "sequential": {
                    "explanation": seq_text
                }
            }
            
            # Add contextual explanation if context is used
            if self.use_context:
                explanation["context"] = {
                    "time_of_day": time_of_day,
                    "device_type": device_type,
                    "explanation": f"Recommended for {time_of_day} viewing on {device_type}"
                }
            
            return explanation
        
        except Exception as e:
            return {"error": f"Error generating explanation: {str(e)}"}
    
    def evaluate_with_ab_testing(self, test_ratio=0.2, with_context=True, without_context=True):
        """
        Perform an A/B testing simulation to evaluate the impact of contextual features.
        
        Args:
            test_ratio (float): Proportion of data to use for testing
            with_context (bool): Whether to evaluate with context
            without_context (bool): Whether to evaluate without context
            
        Returns:
            dict: Dictionary with A/B test results
        """
        # Only run AB testing if both conditions are requested
        if not (with_context and without_context):
            if with_context:
                result = self.evaluate(self.prepare_training_data(test_ratio)[1])
                return {"with_context": {"rmse": result}}
            elif without_context:
                # Temporarily disable context
                old_use_context = self.use_context
                self.use_context = False
                result = self.evaluate(self.prepare_training_data(test_ratio)[1])
                self.use_context = old_use_context
                return {"without_context": {"rmse": result}}
            else:
                return {"error": "No evaluation conditions specified."}
        
        # Prepare data for AB testing
        train_data, test_data = self.prepare_training_data(test_ratio)
        
        # Split test users into two groups
        test_users = test_data['UserID'].unique()
        np.random.seed(42)
        np.random.shuffle(test_users)
        split_point = len(test_users) // 2
        
        group_a_users = test_users[:split_point]  # Without context
        group_b_users = test_users[split_point:]  # With context
        
        group_a_data = test_data[test_data['UserID'].isin(group_a_users)]
        group_b_data = test_data[test_data['UserID'].isin(group_b_users)]
        
        # Evaluate Group A (without context)
        old_use_context = self.use_context
        self.use_context = False
        rmse_a = self.evaluate(group_a_data)
        
        # Evaluate Group B (with context)
        self.use_context = True
        rmse_b = self.evaluate(group_b_data)
        
        # Restore original context setting
        self.use_context = old_use_context
        
        # Calculate relative improvement
        relative_improvement = (rmse_a - rmse_b) / rmse_a * 100
        
        results = {
            "group_a": {
                "name": "Without Context",
                "users": len(group_a_users),
                "samples": len(group_a_data),
                "rmse": rmse_a
            },
            "group_b": {
                "name": "With Context",
                "users": len(group_b_users),
                "samples": len(group_b_data),
                "rmse": rmse_b
            },
            "improvement": {
                "absolute": rmse_a - rmse_b,
                "relative_percent": relative_improvement
            },
            "conclusion": "Contextual features " + 
                         ("improved" if rmse_b < rmse_a else "did not improve") + 
                         " recommendation accuracy."
        }
        
        return results
    
    def save_ab_test_results(self, results, output_file="ab_test_results.json"):
        """
        Save A/B testing results to a JSON file.
        
        Args:
            results (dict): A/B testing results
            output_file (str): Output file path
        """
        # Convert NumPy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Process dictionary recursively
        processed_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                processed_results[k] = {k2: convert_numpy(v2) for k2, v2 in v.items()}
            else:
                processed_results[k] = convert_numpy(v)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        print(f"A/B test results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/processed_movielens_data.csv"
    model_paths = {
        'collaborative': '../models/collaborative_model.pt',
        'content_based': '../models/content_based_model.pt',
        'sequential': '../models/sequential_model.pt'
    }
    
    recommender = HybridRecommender(
        data_path=data_path,
        model_paths=model_paths,
        use_context=True,
        learning_rate=0.001,
        batch_size=128,
        n_epochs=5,
        hybrid_model_path='../models/hybrid_model.pt'
    )
    
    # First ensure base models are trained
    # recommender.cf_model.train()
    # recommender.cb_model.train()
    # recommender.seq_model.train()
    
    # Train hybrid model
    # recommender.train()
    
    # Alternatively, load pre-trained models
    # recommender.load_models()
    
    # Example recommendation
    user_id = 1
    recommendations = recommender.recommend_movies(
        user_id=user_id,
        n=5,
        time_of_day="evening",
        device_type="TV"
    )
    print(f"Hybrid recommendations for user {user_id}:")
    print(recommendations[['MovieID', 'Title', 'HybridScore', 'CF_Contribution', 'CB_Contribution', 'Seq_Contribution']])
    
    # Example explanation
    movie_id = recommendations.iloc[0]['MovieID']
    explanation = recommender.explain_recommendation(user_id, movie_id, "evening", "TV")
    print("\nExplanation:")
    print(json.dumps(explanation, indent=2))
    
    # Run A/B testing
    ab_results = recommender.evaluate_with_ab_testing()
    print("\nA/B Test Results:")
    print(json.dumps(ab_results, indent=2))
    
    # Save results
    recommender.save_ab_test_results(ab_results, "../models/ab_test_results.json")
