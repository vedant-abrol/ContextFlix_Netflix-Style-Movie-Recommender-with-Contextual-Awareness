// App.jsx - Main application component

// Use React Router components
const { BrowserRouter, Routes, Route, Navigate } = ReactRouterDOM;

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
    const API_BASE_URL = 'http://localhost:5000/api';
    
    // Check API health
    React.useEffect(() => {
        const checkApiHealth = async () => {
            try {
                const health = await ApiService.checkHealth();
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
        <BrowserRouter>
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
                        <Route path="/user/:userId/history" element={
                            <UserHistoryPage 
                                {...appContext}
                            />
                        } />
                    </Routes>
                </main>
                
                {/* Movie details modal */}
                {showModal && selectedMovie && (
                    <MovieModal 
                        movie={selectedMovie} 
                        onClose={handleCloseModal}
                        currentUser={currentUser}
                        timeOfDay={timeOfDay}
                        deviceType={deviceType}
                        apiBaseUrl={API_BASE_URL}
                        apiAvailable={apiAvailable}
                    />
                )}
            </div>
        </BrowserRouter>
    );
};

// HomePage Component
const HomePage = ({ currentUser, timeOfDay, deviceType, apiBaseUrl, apiAvailable, handleMovieSelect }) => {
    const [recommendations, setRecommendations] = React.useState([]);
    const [genreRecommendations, setGenreRecommendations] = React.useState({});
    const [loading, setLoading] = React.useState(true);
    
    React.useEffect(() => {
        const fetchRecommendations = async () => {
            try {
                setLoading(true);
                
                if (apiAvailable) {
                    // Try to fetch from API
                    try {
                        const data = await ApiService.getRecommendations(
                            currentUser.id,
                            timeOfDay,
                            deviceType
                        );
                        
                        setRecommendations(data.recommendations);
                        
                        // Fetch genre-based recommendations
                        getGenreRecommendations();
                    } catch (apiError) {
                        console.error('Error fetching recommendations from API:', apiError);
                        useMockData();
                    }
                } else {
                    useMockData();
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error in recommendations fetch process:', err);
                setLoading(false);
            }
        };
        
        const useMockData = () => {
            console.log('Using mock recommendation data');
            const mockData = ApiService.getMockRecommendations(currentUser.id);
            setRecommendations(mockData.recommendations);
            
            // Mock genre recommendations
            setGenreRecommendations({
                'Action': mockData.recommendations.slice(0, 3),
                'Comedy': mockData.recommendations.slice(1, 4),
                'Drama': mockData.recommendations.slice(2, 5)
            });
        };
        
        if (currentUser) {
            fetchRecommendations();
        }
    }, [currentUser, timeOfDay, deviceType, apiAvailable]);
    
    const getGenreRecommendations = () => {
        // Fetch user's preferred genres
        const fetchGenreRecommendations = async () => {
            try {
                // Get user preferences
                const preferences = await ApiService.getUserPreferences(currentUser.id);
                const genres = preferences.preferred_genres;
                
                // Create a genre recommendations object
                const genreRecs = {};
                
                // Get recommendations for each genre (limit to top 3 genres)
                const topGenres = genres.slice(0, 3);
                
                for (const genre of topGenres) {
                    try {
                        // Here we would ideally have an API endpoint that returns recommendations by genre
                        // For now, we'll filter the existing recommendations
                        genreRecs[genre] = recommendations.filter(movie => 
                            movie.genres && movie.genres.includes(genre)
                        ).slice(0, 5);
                        
                        // If we don't have enough movies for this genre, we can mock some
                        if (genreRecs[genre].length < 3) {
                            // Add some mock data to fill in
                            const mockMovies = ApiService.getMockRecommendations(currentUser.id).recommendations;
                            
                            // Assign the genre to these mock movies
                            mockMovies.forEach(movie => {
                                if (!movie.genres.includes(genre)) {
                                    movie.genres.push(genre);
                                }
                            });
                            
                            genreRecs[genre] = [...genreRecs[genre], ...mockMovies].slice(0, 5);
                        }
                    } catch (error) {
                        console.error(`Error fetching recommendations for genre ${genre}:`, error);
                    }
                }
                
                setGenreRecommendations(genreRecs);
            } catch (error) {
                console.error('Error fetching genre recommendations:', error);
            }
        };
        
        fetchGenreRecommendations();
    };
    
    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    return (
        <div className="text-white pb-20">
            {/* Hero Banner with top recommendation */}
            {recommendations.length > 0 && (
                <HeroBanner 
                    movie={recommendations[0]} 
                    onPlay={() => handleMovieSelect(recommendations[0])}
                    onMoreInfo={() => handleMovieSelect(recommendations[0])}
                />
            )}
            
            {/* Main recommendations */}
            <div className="mt-6 px-8">
                <h2 className="text-xl font-bold mb-4">Recommended for You</h2>
                <MovieRow 
                    movies={recommendations} 
                    onMovieSelect={handleMovieSelect}
                />
                
                {/* Contextual message */}
                <div className="mt-4 text-sm text-netflix-light-gray">
                    <p>Recommendations based on your viewing history, time of day ({timeOfDay}), and device type ({deviceType}).</p>
                </div>
                
                {/* Genre-based recommendations */}
                {Object.entries(genreRecommendations).map(([genre, movies]) => (
                    <div key={genre} className="mt-10">
                        <h2 className="text-xl font-bold mb-4">Because You Like {genre}</h2>
                        <MovieRow 
                            movies={movies} 
                            onMovieSelect={handleMovieSelect}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};

// User Selection Page
const UserSelectionPage = ({ users, onUserSelect }) => {
    return (
        <div className="flex flex-col items-center p-8 pt-20">
            <h1 className="text-3xl font-bold mb-10 text-netflix-red">Who's Watching?</h1>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
                {users.map(user => (
                    <div 
                        key={user.id}
                        className="flex flex-col items-center cursor-pointer transition-transform hover:scale-110"
                        onClick={() => onUserSelect(user)}
                    >
                        <div className="w-24 h-24 rounded-md bg-netflix-red flex items-center justify-center text-3xl font-bold mb-3">
                            {user.gender === 'M' ? '♂' : '♀'}
                        </div>
                        <span className="text-gray-300">User {user.id}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// User History Page
const UserHistoryPage = ({ currentUser, apiBaseUrl, apiAvailable }) => {
    const [history, setHistory] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    
    React.useEffect(() => {
        const fetchHistory = async () => {
            try {
                setLoading(true);
                
                if (apiAvailable && currentUser) {
                    // Try to fetch from API
                    try {
                        const data = await ApiService.getUserHistory(currentUser.id);
                        setHistory(data.history || []);
                    } catch (apiError) {
                        console.error('Error fetching history from API:', apiError);
                        useMockHistory();
                    }
                } else {
                    useMockHistory();
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error in history fetch process:', err);
                setLoading(false);
            }
        };
        
        const useMockHistory = () => {
            console.log('Using mock history data');
            // Create some mock history data
            const mockHistory = Array.from({ length: 10 }, (_, i) => ({
                movie_id: i + 1,
                title: `Sample Movie ${i + 1}`,
                rating: Math.floor(Math.random() * 5) + 1,
                timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
                genres: ['Action', 'Comedy', 'Drama'][Math.floor(Math.random() * 3)],
                poster: `https://via.placeholder.com/150x225.png?text=${i+1}`
            }));
            
            setHistory(mockHistory);
        };
        
        if (currentUser) {
            fetchHistory();
        }
    }, [currentUser, apiAvailable]);
    
    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    return (
        <div className="p-8 pt-20">
            <h1 className="text-2xl font-bold mb-8">Viewing History</h1>
            
            <UserHistory history={history} />
        </div>
    );
}; 