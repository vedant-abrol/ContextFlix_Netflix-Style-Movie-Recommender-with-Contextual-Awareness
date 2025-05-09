// App.jsx - Main application component

// Use React Router components
const { HashRouter, Routes, Route, Navigate } = ReactRouterDOM;

// Access all components from window object to ensure they're loaded
const Navbar = window.Navbar;
const MovieCard = window.MovieCard;
const MovieRow = window.MovieRow;
const HeroBanner = window.HeroBanner;
const MovieModal = window.MovieModal;
const UserProfile = window.UserProfile;
const ContextSelector = window.ContextSelector;
const Explanation = window.Explanation;
const UserHistory = window.UserHistory;
const GenreList = window.GenreList;
const SearchBox = window.SearchBox;
const ApiService = window.ApiService;

console.log('Component references:', {
    Navbar, MovieCard, MovieRow, HeroBanner, MovieModal, 
    UserProfile, ContextSelector, Explanation, UserHistory,
    GenreList, SearchBox, ApiService
});

// Main App component
const App = () => {
    // State for the current user, context, and app state
    const [currentUser, setCurrentUser] = React.useState(null);
    const [users, setUsers] = React.useState([]);
    const [timeOfDay, setTimeOfDay] = React.useState('');
    const [deviceType, setDeviceType] = React.useState('');
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);
    const [selectedMovie, setSelectedMovie] = React.useState(null);
    const [showModal, setShowModal] = React.useState(false);
    const [apiAvailable, setApiAvailable] = React.useState(true);
    
    // API base URL - change this to your Flask API endpoint
    const API_BASE_URL = 'http://localhost:5050/api';
    
    console.log('App initialized - Starting health check...');
    
    // Check API health
    React.useEffect(() => {
        const checkApiHealth = async () => {
            try {
                console.log('Checking API health...');
                const health = await ApiService.checkHealth();
                console.log('API health response:', health);
                setApiAvailable(health.status === 'healthy');
            } catch (error) {
                console.error('API health check failed:', error);
                setApiAvailable(false);
            }
        };
        
        checkApiHealth();
    }, []);
    
    // Detect default time of day based on current time
    React.useEffect(() => {
        const hour = new Date().getHours();
        let defaultTime;
        
        if (hour >= 5 && hour < 12) {
            defaultTime = 'morning';
        } else if (hour >= 12 && hour < 17) {
            defaultTime = 'afternoon';
        } else if (hour >= 17 && hour < 21) {
            defaultTime = 'evening';
        } else {
            defaultTime = 'night';
        }
        
        setTimeOfDay(defaultTime);
    }, []);
    
    // Detect default device type based on screen size
    React.useEffect(() => {
        const detectDeviceType = () => {
            const width = window.innerWidth;
            if (width <= 768) {
                return 'mobile';
            } else if (width <= 1024) {
                return 'tablet';
            } else {
                return 'desktop';
            }
        };
        
        setDeviceType(detectDeviceType());
        
        // Update on resize
        window.addEventListener('resize', () => {
            setDeviceType(detectDeviceType());
        });
        
        // Cleanup
        return () => {
            window.removeEventListener('resize', detectDeviceType);
        };
    }, []);
    
    // Fetch available users
    React.useEffect(() => {
        const fetchUsers = async () => {
            try {
                setLoading(true);
                
                if (apiAvailable) {
                    // Try to fetch from API
                    try {
                        const fetchedUsers = await ApiService.getUsers();
                        setUsers(fetchedUsers);
                        
                        // Set first user as default if available
                        if (fetchedUsers && fetchedUsers.length > 0) {
                            setCurrentUser(fetchedUsers[0]);
                        }
                    } catch (apiError) {
                        console.error('Error fetching users from API:', apiError);
                        fallbackToMockUsers();
                    }
                } else {
                    fallbackToMockUsers();
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error in user fetch process:', err);
                setError('Failed to load users. Please check the API server.');
                setLoading(false);
            }
        };
        
        const fallbackToMockUsers = () => {
            console.log('Using mock user data');
            // Create mock users
            const mockUsers = Array.from({ length: 5 }, (_, i) => ({
                id: i + 1,
                gender: i % 2 === 0 ? 'M' : 'F',
                rating_count: Math.floor(Math.random() * 100) + 20,
                avg_rating: (Math.random() * 2) + 3
            }));
            
            setUsers(mockUsers);
            setCurrentUser(mockUsers[0]);
            setApiAvailable(false);
        };
        
        fetchUsers();
    }, [apiAvailable]);
    
    // Handle user selection
    const handleUserSelect = (user) => {
        setCurrentUser(user);
    };
    
    // Handle context changes
    const handleContextChange = (type, value) => {
        if (type === 'time') {
            setTimeOfDay(value);
        } else if (type === 'device') {
            setDeviceType(value);
        }
    };
    
    // Handle movie selection and modal display
    const handleMovieSelect = (movie) => {
        setSelectedMovie(movie);
        setShowModal(true);
    };
    
    // Close the movie details modal
    const handleCloseModal = () => {
        setShowModal(false);
    };
    
    // Context object to provide to child components
    const appContext = {
        currentUser,
        users,
        timeOfDay,
        deviceType,
        apiBaseUrl: API_BASE_URL,
        apiAvailable,
        handleUserSelect,
        handleContextChange,
        handleMovieSelect,
        handleCloseModal
    };
    
    // Loading state
    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    // Error state
    if (error) {
        return (
            <div className="flex flex-col items-center justify-center h-screen p-4">
                <div className="text-netflix-red text-2xl mb-4">Error</div>
                <div className="mb-4">{error}</div>
                <button 
                    className="netflix-button"
                    onClick={() => window.location.reload()}
                >
                    Retry
                </button>
            </div>
        );
    }
    
    return (
        <HashRouter>
            <div className="app-container">
                {/* API status warning */}
                {!apiAvailable && (
                    <div className="bg-yellow-600 text-white text-center p-2 text-sm">
                        API server not available. Using mock data instead.
                    </div>
                )}
                
                {/* Navbar */}
                <Navbar 
                    currentUser={currentUser} 
                    users={users} 
                    onUserSelect={handleUserSelect}
                    timeOfDay={timeOfDay}
                    deviceType={deviceType}
                    onContextChange={handleContextChange}
                />
                
                {/* Main content */}
                <main>
                    <Routes>
                        <Route path="/" element={
                            currentUser ? (
                                <HomePage 
                                    {...appContext}
                                />
                            ) : (
                                <Navigate to="/select-user" />
                            )
                        } />
                        
                        <Route path="/select-user" element={
                            <UserSelectionPage 
                                users={users} 
                                onUserSelect={handleUserSelect} 
                            />
                        } />
                        
                        <Route path="/history" element={
                            currentUser ? (
                                <UserHistoryPage 
                                    currentUser={currentUser}
                                    apiBaseUrl={API_BASE_URL}
                                    apiAvailable={apiAvailable}
                                />
                            ) : (
                                <Navigate to="/select-user" />
                            )
                        } />
                    </Routes>
                </main>
                
                {/* Movie details modal */}
                {showModal && selectedMovie && (
                    <MovieModal 
                        movie={selectedMovie} 
                        onClose={handleCloseModal}
                        currentUser={currentUser}
                        apiBaseUrl={API_BASE_URL}
                    />
                )}
                
                {/* Footer */}
                <footer className="p-6 text-center text-netflix-light-gray">
                    <p>ContextFlix - Personalized Movie Recommender</p>
                    <p className="text-sm mt-2">© 2023 - Movie data from MovieLens</p>
                </footer>
            </div>
        </HashRouter>
    );
};

// HomePage component
const HomePage = ({ currentUser, timeOfDay, deviceType, apiBaseUrl, apiAvailable, handleMovieSelect }) => {
    const [recommendations, setRecommendations] = React.useState([]);
    const [topRatedMovies, setTopRatedMovies] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);
    
    // Fetch recommendations when user or context changes
    React.useEffect(() => {
        const fetchRecommendations = async () => {
            if (!currentUser) return;
            
            try {
                setLoading(true);
                
                if (apiAvailable) {
                    try {
                        // Get personalized recommendations from API
                        const recResponse = await ApiService.getRecommendations(
                            currentUser.id, timeOfDay, deviceType
                        );
                        
                        setRecommendations(recResponse.recommendations);
                        
                        // Get some general top-rated movies
                        const moviesResponse = await axios.get(`${apiBaseUrl}/movies?limit=20`);
                        setTopRatedMovies(moviesResponse.data.movies);
                    } catch (apiError) {
                        console.error('API error, falling back to mock data:', apiError);
                        useMockData();
                    }
                } else {
                    useMockData();
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error fetching recommendations:', err);
                setError('Failed to load recommendations');
                setLoading(false);
            }
        };
        
        const useMockData = () => {
            // Use mock data when API is not available
            const mockRecs = ApiService.getMockRecommendations(currentUser.id);
            setRecommendations(mockRecs.recommendations);
            
            // Create mock top-rated movies
            const mockTopRated = Array.from({ length: 10 }, (_, i) => ({
                MovieID: 100 + i,
                Title: `Popular Movie ${i+1} (${2010 + i})`,
                Genres: 'Action|Adventure|Sci-Fi',
                poster: `https://via.placeholder.com/150x225.png?text=${100+i}`
            }));
            
            setTopRatedMovies(mockTopRated);
        };
        
        fetchRecommendations();
    }, [currentUser, timeOfDay, deviceType, apiBaseUrl, apiAvailable]);
    
    // Group recommendations by their primary genre
    const getGenreRecommendations = () => {
        const genreGroups = {};
        
        recommendations.forEach(movie => {
            if (movie.genres && movie.genres.length > 0) {
                const primaryGenre = movie.genres[0];
                
                if (!genreGroups[primaryGenre]) {
                    genreGroups[primaryGenre] = [];
                }
                
                genreGroups[primaryGenre].push(movie);
            }
        });
        
        return Object.entries(genreGroups)
            .filter(([_, movies]) => movies.length >= 3) // Only show genres with at least 3 movies
            .sort((a, b) => b[1].length - a[1].length); // Sort by number of movies
    };
    
    // Display loading state
    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    // Display error state
    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen p-4">
                <div className="text-netflix-red text-xl mb-4">Error</div>
                <div className="mb-4">{error}</div>
                <button 
                    className="netflix-button"
                    onClick={() => window.location.reload()}
                >
                    Retry
                </button>
            </div>
        );
    }
    
    // Get featured movie for hero banner (first recommended movie)
    const featuredMovie = recommendations.length > 0 ? recommendations[0] : null;
    
    // Get genre-based recommendations
    const genreRecommendations = getGenreRecommendations();
    
    return (
        <div className="home-page">
            {/* Hero banner with featured movie */}
            {featuredMovie && (
                <HeroBanner 
                    movie={featuredMovie} 
                    onSelect={() => handleMovieSelect(featuredMovie)}
                />
            )}
            
            <div className="recommendations-container pt-8">
                {/* Context-aware recommendations */}
                <MovieRow 
                    title={`Recommended for ${currentUser.id} on ${timeOfDay} (${deviceType})`}
                    movies={recommendations}
                    onMovieSelect={handleMovieSelect}
                />
                
                {/* Genre-based recommendations */}
                {genreRecommendations.map(([genre, movies]) => (
                    <MovieRow 
                        key={genre}
                        title={`${genre} Movies For You`}
                        movies={movies}
                        onMovieSelect={handleMovieSelect}
                    />
                ))}
                
                {/* Popular movies */}
                <MovieRow 
                    title="Popular on ContextFlix"
                    movies={topRatedMovies}
                    onMovieSelect={handleMovieSelect}
                />
            </div>
        </div>
    );
};

// UserSelectionPage component
const UserSelectionPage = ({ users, onUserSelect }) => {
    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-4">
            <h1 className="text-4xl font-bold mb-8 text-netflix-red">Who's Watching?</h1>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {users.map(user => (
                    <div 
                        key={user.id}
                        className="flex flex-col items-center p-4 cursor-pointer transition-transform hover:scale-110"
                        onClick={() => onUserSelect(user)}
                    >
                        <div className="w-24 h-24 rounded-md bg-netflix-red flex items-center justify-center text-4xl mb-2">
                            {user.id}
                        </div>
                        <span className="text-lg text-center">
                            User {user.id}
                            {user.gender && ` (${user.gender})`}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// UserHistoryPage component
const UserHistoryPage = ({ currentUser, apiBaseUrl, apiAvailable }) => {
    const [history, setHistory] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);
    
    React.useEffect(() => {
        const fetchHistory = async () => {
            if (!currentUser) return;
            
            try {
                setLoading(true);
                
                if (apiAvailable) {
                    try {
                        const response = await ApiService.getUserHistory(currentUser.id);
                        setHistory(response.history);
                    } catch (apiError) {
                        console.error('API error, using mock history data:', apiError);
                        useMockHistory();
                    }
                } else {
                    useMockHistory();
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error fetching user history:', err);
                setError('Failed to load user history');
                setLoading(false);
            }
        };
        
        const useMockHistory = () => {
            // Create mock history data
            const mockHistory = Array.from({ length: 8 }, (_, i) => ({
                movie_id: i + 10,
                title: `Movie ${i+1} (${2020 - i})`,
                rating: 3.5 + Math.random() * 1.5,
                poster: `https://via.placeholder.com/150x225.png?text=${i+10}`,
                time_of_day: ['morning', 'afternoon', 'evening', 'night'][i % 4],
                device_type: ['mobile', 'tablet', 'desktop', 'TV'][i % 4],
                genres: ['Action', 'Comedy', 'Drama', 'Horror'][i % 4].split('|'),
                timestamp: new Date(Date.now() - i * 86400000).toISOString()
            }));
            
            setHistory(mockHistory);
        };
        
        fetchHistory();
    }, [currentUser, apiBaseUrl, apiAvailable]);
    
    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen p-4">
                <div className="text-netflix-red text-xl mb-4">Error</div>
                <div className="mb-4">{error}</div>
                <button 
                    className="netflix-button"
                    onClick={() => window.location.reload()}
                >
                    Retry
                </button>
            </div>
        );
    }
    
    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-6">Viewing History</h1>
            
            {history.length === 0 ? (
                <p>No viewing history available.</p>
            ) : (
                <UserHistory history={history} />
            )}
        </div>
    );
};

// Export App component to make it globally available
window.App = App;
