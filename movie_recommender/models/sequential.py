import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class TransformerRecommender(nn.Module):
    def __init__(self, n_movies, embedding_dim=64, n_heads=4, n_layers=2, dropout=0.1, max_seq_len=50):
        """
        Transformer-based sequential recommendation model.
        
        Args:
            n_movies (int): Number of unique movies
            embedding_dim (int): Dimension of embeddings
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length of user history
        """
        super(TransformerRecommender, self).__init__()
        
        # Movie embedding layer (add +1 for padding token at index 0)
        self.movie_embeddings = nn.Embedding(n_movies + 1, embedding_dim, padding_idx=0)
        
        # Context embeddings for time_of_day and device_type
        self.time_embeddings = nn.Embedding(4, embedding_dim // 4)  # 4 time categories
        self.device_embeddings = nn.Embedding(4, embedding_dim // 4)  # 4 device types
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, n_movies + 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.movie_embeddings.weight, mean=0, std=0.01)
        nn.init.normal_(self.time_embeddings.weight, mean=0, std=0.01)
        nn.init.normal_(self.device_embeddings.weight, mean=0, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, seq, times=None, devices=None, mask=None):
        """
        Forward pass of the model.
        
        Args:
            seq (torch.Tensor): Batch of movie ID sequences [batch_size, seq_len]
            times (torch.Tensor, optional): Batch of time_of_day sequences [batch_size, seq_len]
            devices (torch.Tensor, optional): Batch of device_type sequences [batch_size, seq_len]
            mask (torch.Tensor, optional): Mask for padding positions [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Predicted next movie scores [batch_size, n_movies]
        """
        # Get embeddings
        seq_emb = self.movie_embeddings(seq)  # [batch_size, seq_len, embedding_dim]
        
        # Add contextual embeddings if provided
        if times is not None:
            time_emb = self.time_embeddings(times)
            # Resize to match embedding_dim by repeating
            time_emb = time_emb.repeat(1, 1, 4)
            seq_emb = seq_emb + time_emb
            
        if devices is not None:
            device_emb = self.device_embeddings(devices)
            # Resize to match embedding_dim by repeating
            device_emb = device_emb.repeat(1, 1, 4)
            seq_emb = seq_emb + device_emb
        
        # Add positional encoding
        seq_emb = self.pos_encoder(seq_emb)
        
        # Apply transformer encoder
        if mask is not None:
            # Transformer expects 1 for valid positions and 0 for masked positions
            transformer_mask = ~mask  # Invert the mask
            output = self.transformer_encoder(seq_emb, src_key_padding_mask=transformer_mask)
        else:
            output = self.transformer_encoder(seq_emb)
        
        # Take the last sequence element for prediction
        last_item_hidden = output[:, -1, :]
        
        # Project to vocabulary size
        logits = self.output_layer(last_item_hidden)
        
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding module for transformer.
        
        Args:
            d_model (int): Embedding dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MovieSequenceDataset(Dataset):
    def __init__(self, sequences, targets, time_seqs=None, device_seqs=None, max_len=50):
        """
        Dataset for sequential movie recommendation.
        
        Args:
            sequences (list): List of movie ID sequences
            targets (list): List of target movie IDs
            time_seqs (list, optional): List of time_of_day sequences
            device_seqs (list, optional): List of device_type sequences
            max_len (int): Maximum sequence length
        """
        self.sequences = sequences
        self.targets = targets
        self.time_seqs = time_seqs
        self.device_seqs = device_seqs
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]  # Take last max_len items
        target = self.targets[idx]
        
        # Pad sequence if needed
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            seq = [0] * padding_len + seq  # Pad with zeros
        
        # Create mask (1 for valid positions, 0 for padding)
        mask = [0] * padding_len + [1] * min(len(self.sequences[idx]), self.max_len)
        
        # Convert to tensors
        seq_tensor = torch.LongTensor(seq)
        target_tensor = torch.LongTensor([target])
        mask_tensor = torch.BoolTensor(mask)
        
        # Add contextual features if available
        if self.time_seqs is not None and self.device_seqs is not None:
            time_seq = self.time_seqs[idx][-self.max_len:]
            device_seq = self.device_seqs[idx][-self.max_len:]
            
            # Pad time and device sequences
            if padding_len > 0:
                time_seq = [0] * padding_len + time_seq
                device_seq = [0] * padding_len + device_seq
            
            time_tensor = torch.LongTensor(time_seq)
            device_tensor = torch.LongTensor(device_seq)
            
            return {'seq': seq_tensor, 'time': time_tensor, 'device': device_tensor, 
                    'mask': mask_tensor, 'target': target_tensor}
        
        return {'seq': seq_tensor, 'mask': mask_tensor, 'target': target_tensor}


class SequentialModel:
    def __init__(self, data_path=None, embedding_dim=64, n_heads=4, n_layers=2, 
                 dropout=0.1, max_seq_len=50, learning_rate=0.001, batch_size=64, 
                 n_epochs=10, model_path=None, device=None, use_context=True):
        """
        Sequential movie recommendation model using transformer.
        
        Args:
            data_path (str): Path to the processed dataset
            embedding_dim (int): Dimension of embeddings
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            model_path (str): Path to save/load model weights
            device (str): Device to use for training ('cuda' or 'cpu')
            use_context (bool): Whether to use contextual features
        """
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_path = model_path or 'sequential_model.pt'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_context = use_context
        
        self.model = None
        self.movie_encoder = None
        self.time_encoder = None
        self.device_encoder = None
        self.movie_decoder = None
        self.time_decoder = None
        self.device_decoder = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load and preprocess data for sequential recommendation.
        
        Args:
            data_path (str): Path to the processed dataset
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Create encoders for categorical features
        self.movie_encoder = LabelEncoder()
        self.data['MovieIdx'] = self.movie_encoder.fit_transform(self.data['MovieID']) + 1  # +1 for padding token at 0
        
        if self.use_context:
            self.time_encoder = LabelEncoder()
            self.device_encoder = LabelEncoder()
            self.data['TimeIdx'] = self.time_encoder.fit_transform(self.data['Time_of_Day']) + 1
            self.data['DeviceIdx'] = self.device_encoder.fit_transform(self.data['Device_Type']) + 1
        
        # Create decoders (for mapping back to original IDs)
        self.movie_decoder = {i+1: movie_id for i, movie_id in enumerate(self.movie_encoder.classes_)}
        
        if self.use_context:
            self.time_decoder = {i+1: time for i, time in enumerate(self.time_encoder.classes_)}
            self.device_decoder = {i+1: device for i, device in enumerate(self.device_encoder.classes_)}
        
        n_movies = len(self.movie_encoder.classes_)
        print(f"Processed data: {len(self.data)} ratings, {n_movies} unique movies")
        
        # Initialize model
        self.model = TransformerRecommender(
            n_movies=n_movies,
            embedding_dim=self.embedding_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len
        ).to(self.device)
    
    def prepare_sequences(self, min_seq_len=3, val_ratio=0.1):
        """
        Prepare sequence data for training and validation.
        
        Args:
            min_seq_len (int): Minimum sequence length to include
            val_ratio (float): Proportion of data to use for validation
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Sort data by user and timestamp
        self.data = self.data.sort_values(by=['UserID', 'Timestamp'])
        
        # Group data by user
        user_sequences = defaultdict(list)
        user_time_sequences = defaultdict(list) if self.use_context else None
        user_device_sequences = defaultdict(list) if self.use_context else None
        
        for _, row in self.data.iterrows():
            user_id = row['UserID']
            movie_idx = row['MovieIdx']
            user_sequences[user_id].append(movie_idx)
            
            if self.use_context:
                time_idx = row['TimeIdx']
                device_idx = row['DeviceIdx']
                user_time_sequences[user_id].append(time_idx)
                user_device_sequences[user_id].append(device_idx)
        
        # Create training sequences and targets
        sequences = []
        targets = []
        time_sequences = [] if self.use_context else None
        device_sequences = [] if self.use_context else None
        
        for user_id, user_seq in user_sequences.items():
            if len(user_seq) >= min_seq_len:
                for i in range(min_seq_len - 1, len(user_seq) - 1):
                    sequences.append(user_seq[:i+1])
                    targets.append(user_seq[i+1])
                    
                    if self.use_context:
                        time_sequences.append(user_time_sequences[user_id][:i+1])
                        device_sequences.append(user_device_sequences[user_id][:i+1])
        
        # Split into train and validation
        n_val = int(len(sequences) * val_ratio)
        indices = list(range(len(sequences)))
        random.seed(42)
        random.shuffle(indices)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_sequences = [sequences[i] for i in train_indices]
        train_targets = [targets[i] for i in train_indices]
        
        val_sequences = [sequences[i] for i in val_indices]
        val_targets = [targets[i] for i in val_indices]
        
        if self.use_context:
            train_time_sequences = [time_sequences[i] for i in train_indices]
            train_device_sequences = [device_sequences[i] for i in train_indices]
            
            val_time_sequences = [time_sequences[i] for i in val_indices]
            val_device_sequences = [device_sequences[i] for i in val_indices]
            
            train_dataset = MovieSequenceDataset(
                train_sequences, train_targets, train_time_sequences, train_device_sequences, self.max_seq_len
            )
            val_dataset = MovieSequenceDataset(
                val_sequences, val_targets, val_time_sequences, val_device_sequences, self.max_seq_len
            )
        else:
            train_dataset = MovieSequenceDataset(train_sequences, train_targets, max_len=self.max_seq_len)
            val_dataset = MovieSequenceDataset(val_sequences, val_targets, max_len=self.max_seq_len)
        
        print(f"Created {len(train_dataset)} training sequences and {len(val_dataset)} validation sequences")
        
        return train_dataset, val_dataset
    
    def train(self, save_model=True):
        """
        Train the sequential model.
        
        Args:
            save_model (bool): Whether to save the model after training
            
        Returns:
            tuple: (train_losses, val_losses, hitrates)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_sequences()
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        hitrates = []
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                seq = batch['seq'].to(self.device)
                mask = batch['mask'].to(self.device)
                target = batch['target'].to(self.device).squeeze()
                
                if self.use_context and 'time' in batch and 'device' in batch:
                    time_seq = batch['time'].to(self.device)
                    device_seq = batch['device'].to(self.device)
                    outputs = self.model(seq, time_seq, device_seq, mask)
                else:
                    outputs = self.model(seq, mask=mask)
                
                loss = criterion(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            hit_count = 0
            total_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    seq = batch['seq'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    target = batch['target'].to(self.device).squeeze()
                    
                    if self.use_context and 'time' in batch and 'device' in batch:
                        time_seq = batch['time'].to(self.device)
                        device_seq = batch['device'].to(self.device)
                        outputs = self.model(seq, time_seq, device_seq, mask)
                    else:
                        outputs = self.model(seq, mask=mask)
                    
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    
                    # Calculate hit rate@10
                    _, top_indices = torch.topk(outputs, 10)
                    hits = (top_indices == target.unsqueeze(1)).any(dim=1).sum().item()
                    hit_count += hits
                    total_count += target.size(0)
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            hitrate = hit_count / total_count
            hitrates.append(hitrate)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.n_epochs} | " 
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | " 
                  f"HitRate@10: {hitrate:.4f}")
        
        if save_model:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
        
        return train_losses, val_losses, hitrates
    
    def load_model(self):
        """
        Load a trained model from disk.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_data() first.")
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def get_user_sequence(self, user_id, max_items=None):
        """
        Get a user's viewing sequence, sorted by timestamp.
        
        Args:
            user_id (int): User ID
            max_items (int, optional): Maximum number of items to include
            
        Returns:
            list: List of movie indices
        """
        user_data = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')
        
        if len(user_data) == 0:
            raise ValueError(f"User ID {user_id} not found in training data.")
        
        if max_items:
            user_data = user_data.tail(max_items)
        
        movie_sequence = user_data['MovieIdx'].tolist()
        
        if self.use_context:
            time_sequence = user_data['TimeIdx'].tolist()
            device_sequence = user_data['DeviceIdx'].tolist()
            return movie_sequence, time_sequence, device_sequence
        
        return movie_sequence
    
    def recommend_next_movies(self, user_id, n=10, time_of_day=None, device_type=None):
        """
        Recommend next movies for a user based on their viewing history.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            time_of_day (str, optional): Current time of day (morning, afternoon, evening, night)
            device_type (str, optional): Current device type (mobile, tablet, desktop, TV)
            
        Returns:
            pandas.DataFrame: DataFrame with movie recommendations
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")
        
        # Get user sequence
        if self.use_context:
            movie_seq, time_seq, device_seq = self.get_user_sequence(user_id, self.max_seq_len)
        else:
            movie_seq = self.get_user_sequence(user_id, self.max_seq_len)
        
        # Check if sequence is too short
        if len(movie_seq) < 2:
            raise ValueError(f"User {user_id} has too few ratings for sequential recommendation.")
        
        # Convert to tensor
        seq_tensor = torch.LongTensor([movie_seq[-self.max_seq_len:]]).to(self.device)
        
        # Create mask
        if len(movie_seq) < self.max_seq_len:
            padding_len = self.max_seq_len - len(movie_seq)
            mask = torch.BoolTensor([[0] * padding_len + [1] * len(movie_seq)]).to(self.device)
            
            # Pad sequence
            padded_seq = torch.zeros(1, self.max_seq_len, dtype=torch.long).to(self.device)
            padded_seq[0, -len(movie_seq):] = seq_tensor
            seq_tensor = padded_seq
        else:
            mask = torch.ones(1, self.max_seq_len, dtype=torch.bool).to(self.device)
        
        # Prepare context if needed
        if self.use_context:
            # Handle time sequences
            if len(time_seq) < self.max_seq_len:
                padding_len = self.max_seq_len - len(time_seq)
                padded_time = torch.zeros(1, self.max_seq_len, dtype=torch.long).to(self.device)
                padded_time[0, -len(time_seq):] = torch.LongTensor([time_seq]).to(self.device)
                time_tensor = padded_time
            else:
                time_tensor = torch.LongTensor([time_seq[-self.max_seq_len:]]).to(self.device)
            
            # Handle device sequences
            if len(device_seq) < self.max_seq_len:
                padding_len = self.max_seq_len - len(device_seq)
                padded_device = torch.zeros(1, self.max_seq_len, dtype=torch.long).to(self.device)
                padded_device[0, -len(device_seq):] = torch.LongTensor([device_seq]).to(self.device)
                device_tensor = padded_device
            else:
                device_tensor = torch.LongTensor([device_seq[-self.max_seq_len:]]).to(self.device)
            
            # Update last time and device if provided
            if time_of_day and device_type:
                try:
                    time_idx = self.time_encoder.transform([time_of_day])[0] + 1
                    device_idx = self.device_encoder.transform([device_type])[0] + 1
                    
                    time_tensor[0, -1] = time_idx
                    device_tensor[0, -1] = device_idx
                except:
                    print(f"Warning: Invalid time_of_day '{time_of_day}' or device_type '{device_type}'")
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(seq_tensor, time_tensor, device_tensor, mask)
        else:
            # Get predictions without context
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(seq_tensor, mask=mask)
        
        # Zero out scores for movies in the sequence (exclude already watched)
        for movie_idx in movie_seq:
            outputs[0, movie_idx] = float('-inf')
        
        # Get top N recommendations
        _, top_indices = torch.topk(outputs[0], n)
        top_indices = top_indices.cpu().numpy()
        
        # Convert indices to original movie IDs
        movie_ids = [self.movie_decoder[idx.item()] for idx in top_indices]
        scores = outputs[0, top_indices].cpu().numpy()
        
        # Get movie titles
        movie_data = self.data[['MovieID', 'Title']].drop_duplicates()
        movie_data = movie_data[movie_data['MovieID'].isin(movie_ids)]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'MovieID': movie_ids,
            'Score': scores
        })
        
        recommendations = pd.merge(recommendations, movie_data, on='MovieID', how='left')
        
        # Add context info if provided
        if time_of_day and device_type and self.use_context:
            recommendations['Time_of_Day'] = time_of_day
            recommendations['Device_Type'] = device_type
        
        return recommendations[['MovieID', 'Title', 'Score']]
    
    def explain_sequential_recommendation(self, user_id, recommended_movie_id):
        """
        Explain why a movie was recommended based on sequential patterns.
        
        Args:
            user_id (int): User ID
            recommended_movie_id (int): Recommended movie ID
            
        Returns:
            dict: Explanation dictionary
        """
        # Get user's recent movies
        user_data = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')
        
        if len(user_data) < 3:
            return {"explanation": "Not enough viewing history for sequential explanation."}
        
        # Get the 5 most recent movies
        recent_movies = user_data.tail(5)[['MovieID', 'Title']].values.tolist()
        
        explanation = {
            "recent_movies": [{"id": movie_id, "title": title} for movie_id, title in recent_movies],
            "explanation": f"Based on your recent viewing pattern, especially after watching {recent_movies[-1][1]}.",
            "recommended_after": [recent_movies[-1][1]]
        }
        
        # Add time and device context if available
        if self.use_context:
            last_time = user_data.iloc[-1]['Time_of_Day']
            last_device = user_data.iloc[-1]['Device_Type']
            
            explanation["context"] = {
                "time_of_day": last_time,
                "device_type": last_device
            }
            explanation["explanation"] += f" This was recommended for {last_time} viewing on {last_device}."
        
        return explanation


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/processed_movielens_data.csv"
    model = SequentialModel(
        data_path=data_path,
        embedding_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        max_seq_len=50,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=5,
        model_path="../models/sequential_model.pt",
        use_context=True
    )
    
    # Train model
    train_losses, val_losses, hitrates = model.train()
    
    # Example recommendation
    user_id = 1
    recommendations = model.recommend_next_movies(
        user_id=user_id,
        n=5,
        time_of_day="evening",
        device_type="TV"
    )
    print(f"Next movie recommendations for user {user_id}:")
    print(recommendations)
    
    # Example explanation
    explanation = model.explain_sequential_recommendation(user_id, recommendations.iloc[0]['MovieID'])
    print("\nExplanation:")
    print(explanation)
