// ApiService.jsx - Service for making API calls

const API_BASE_URL = 'http://localhost:5000/api';

const ApiService = {
    // Health check
    checkHealth: async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/health`);
            return response.data;
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unhealthy', error: error.message };
        }
    },
    
    // Get users
    getUsers: async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/users`);
            return response.data.users;
        } catch (error) {
            console.error('Failed to fetch users:', error);
            throw error;
        }
    },
    
    // Get recommendations for a user
    getRecommendations: async (userId, timeOfDay, deviceType, refresh = false) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/recommend`, {
                params: { user_id: userId, time_of_day: timeOfDay, device_type: deviceType, refresh }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch recommendations:', error);
            throw error;
        }
    },
    
    // Get user history
    getUserHistory: async (userId, limit = 20) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/user_history`, {
                params: { user_id: userId, limit }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch user history:', error);
            throw error;
        }
    },
    
    // Get similar movies
    getSimilarMovies: async (movieId, n = 10) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/similar_movies`, {
                params: { movie_id: movieId, n }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch similar movies:', error);
            throw error;
        }
    },
    
    // Get all movie genres
    getGenres: async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/movie_genres`);
            return response.data.genres;
        } catch (error) {
            console.error('Failed to fetch genres:', error);
            throw error;
        }
    },
    
    // Get user preferences (genres they like)
    getUserPreferences: async (userId) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/user_preferences`, {
                params: { user_id: userId }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch user preferences:', error);
            throw error;
        }
    },
    
    // Get A/B test results
    getABTestResults: async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/ab_test_results`);
            return response.data;
        } catch (error) {
            console.error('Failed to fetch A/B test results:', error);
            throw error;
        }
    },
    
    // Fallback method for mock data when API is not available
    getMockRecommendations: (userId) => {
        return {
            user_id: userId,
            time_of_day: 'evening',
            device_type: 'desktop',
            count: 5,
            recommendations: [
                {
                    movie_id: 1,
                    title: 'Toy Story (1995)',
                    score: 4.5,
                    poster: 'https://via.placeholder.com/150x225.png?text=1',
                    genres: ['Animation', "Children's", 'Comedy']
                },
                {
                    movie_id: 2,
                    title: 'Jumanji (1995)',
                    score: 4.2,
                    poster: 'https://via.placeholder.com/150x225.png?text=2',
                    genres: ['Adventure', "Children's", 'Fantasy']
                },
                {
                    movie_id: 3,
                    title: 'Grumpier Old Men (1995)',
                    score: 3.9,
                    poster: 'https://via.placeholder.com/150x225.png?text=3',
                    genres: ['Comedy', 'Romance']
                },
                {
                    movie_id: 4,
                    title: 'Waiting to Exhale (1995)',
                    score: 3.5,
                    poster: 'https://via.placeholder.com/150x225.png?text=4',
                    genres: ['Comedy', 'Drama']
                },
                {
                    movie_id: 5,
                    title: 'Father of the Bride Part II (1995)',
                    score: 3.8,
                    poster: 'https://via.placeholder.com/150x225.png?text=5',
                    genres: ['Comedy']
                }
            ]
        };
    }
}; 