// MovieModal.jsx - Detailed movie information modal with recommendations

const MovieModal = ({ movie, onClose, currentUser, timeOfDay, deviceType, apiBaseUrl }) => {
    const [similarMovies, setSimilarMovies] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const modalRef = React.useRef(null);
    
    // Clean title (remove year in parentheses)
    const cleanTitle = () => {
        if (!movie.title) return '';
        const yearMatch = movie.title.match(/\s*\(\d{4}\)$/);
        if (yearMatch) {
            return movie.title.replace(yearMatch[0], '');
        }
        return movie.title;
    };
    
    // Get year from title or movie object
    const getYear = () => {
        if (movie.year) return movie.year;
        
        if (movie.title && movie.title.includes('(') && movie.title.includes(')')) {
            const startIndex = movie.title.lastIndexOf('(') + 1;
            const endIndex = movie.title.lastIndexOf(')');
            if (startIndex < endIndex) {
                return movie.title.substring(startIndex, endIndex);
            }
        }
        return '';
    };
    
    // Fetch similar movies when modal opens
    React.useEffect(() => {
        const fetchSimilarMovies = async () => {
            try {
                setLoading(true);
                const movieId = movie.movie_id || movie.MovieID;
                const response = await axios.get(`${apiBaseUrl}/similar_movies?movie_id=${movieId}&n=6`);
                setSimilarMovies(response.data.similar_movies || []);
                setLoading(false);
            } catch (err) {
                console.error('Error fetching similar movies:', err);
                setSimilarMovies([]);
                setLoading(false);
            }
        };
        
        fetchSimilarMovies();
        
        // Close modal on escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        
        window.addEventListener('keydown', handleEscape);
        
        return () => {
            window.removeEventListener('keydown', handleEscape);
        };
    }, [movie]);
    
    // Close modal when clicking outside content
    const handleOutsideClick = (e) => {
        if (modalRef.current && !modalRef.current.contains(e.target)) {
            onClose();
        }
    };
    
    // Get recommendation explanation
    const getExplanationSection = () => {
        if (!movie.explanation) return null;
        
        const sections = [];
        
        // Collaborative filtering explanation
        if (movie.explanation.collaborative_filtering) {
            sections.push(
                <div key="cf" className="mb-4">
                    <h4 className="text-lg font-semibold mb-2">Similar Users</h4>
                    <p>{movie.explanation.collaborative_filtering.explanation}</p>
                </div>
            );
        }
        
        // Content-based explanation
        if (movie.explanation.content_based) {
            sections.push(
                <div key="cb" className="mb-4">
                    <h4 className="text-lg font-semibold mb-2">Genre Preferences</h4>
                    <p>{movie.explanation.content_based.explanation}</p>
                    
                    {movie.explanation.content_based.genres && movie.explanation.content_based.genres.length > 0 && (
                        <div className="flex flex-wrap mt-2">
                            {movie.explanation.content_based.genres.map((genre, index) => (
                                <span 
                                    key={index} 
                                    className="bg-gray-800 text-white px-2 py-1 rounded mr-2 mb-2 text-sm"
                                >
                                    {genre.name} ({(genre.importance * 100).toFixed(0)}%)
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            );
        }
        
        // Sequential explanation
        if (movie.explanation.sequential) {
            sections.push(
                <div key="seq" className="mb-4">
                    <h4 className="text-lg font-semibold mb-2">Viewing History</h4>
                    <p>{movie.explanation.sequential.explanation}</p>
                    
                    {movie.explanation.sequential.recent_movies && movie.explanation.sequential.recent_movies.length > 0 && (
                        <div className="mt-2">
                            <p className="text-sm text-gray-400 mb-1">Recently watched:</p>
                            <div className="flex flex-wrap">
                                {movie.explanation.sequential.recent_movies.map((recent, index) => (
                                    <span 
                                        key={index} 
                                        className="bg-gray-800 text-white px-2 py-1 rounded mr-2 mb-2 text-sm"
                                    >
                                        {recent.title}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            );
        }
        
        // Contextual explanation
        if (movie.explanation.context) {
            sections.push(
                <div key="ctx" className="mb-4">
                    <h4 className="text-lg font-semibold mb-2">Viewing Context</h4>
                    <p>{movie.explanation.context.explanation}</p>
                    <div className="flex mt-2">
                        <span className="bg-netflix-red text-white px-2 py-1 rounded mr-2 text-sm">
                            {movie.explanation.context.time_of_day}
                        </span>
                        <span className="bg-netflix-red text-white px-2 py-1 rounded mr-2 text-sm">
                            {movie.explanation.context.device_type}
                        </span>
                    </div>
                </div>
            );
        }
        
        if (sections.length === 0) return null;
        
        return (
            <div className="mt-6 border-t border-gray-700 pt-4">
                <h3 className="text-xl font-bold mb-4">Why This Was Recommended</h3>
                {sections}
            </div>
        );
    };
    
    // Model contribution visualization if available
    const getContributionChart = () => {
        if (!movie.contributions) return null;
        
        const { collaborative, content_based, sequential } = movie.contributions;
        
        const formatPercent = (value) => {
            return `${(value * 100).toFixed(0)}%`;
        };
        
        return (
            <div className="mt-6 border-t border-gray-700 pt-4">
                <h3 className="text-xl font-bold mb-4">Recommendation Factors</h3>
                
                <div className="space-y-3">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span>Collaborative Filtering</span>
                            <span>{formatPercent(collaborative)}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5">
                            <div 
                                className="bg-blue-500 h-2.5 rounded-full" 
                                style={{ width: formatPercent(collaborative) }}
                            ></div>
                        </div>
                    </div>
                    
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span>Content-Based</span>
                            <span>{formatPercent(content_based)}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5">
                            <div 
                                className="bg-green-500 h-2.5 rounded-full" 
                                style={{ width: formatPercent(content_based) }}
                            ></div>
                        </div>
                    </div>
                    
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span>Sequential</span>
                            <span>{formatPercent(sequential)}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5">
                            <div 
                                className="bg-yellow-500 h-2.5 rounded-full" 
                                style={{ width: formatPercent(sequential) }}
                            ></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };
    
    return (
        <div className="movie-modal" onClick={handleOutsideClick}>
            <div 
                ref={modalRef}
                className="movie-modal-content max-w-4xl"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Close button */}
                <button 
                    className="absolute top-4 right-4 text-white hover:text-gray-300 z-10"
                    onClick={onClose}
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
                
                {/* Movie poster and details */}
                <div className="p-6">
                    <div className="flex flex-col md:flex-row">
                        {/* Movie poster */}
                        <div className="flex-shrink-0 w-full md:w-1/3 mb-4 md:mb-0">
                            <img 
                                src={movie.poster || `https://via.placeholder.com/300x450/141414/E50914?text=${movie.movie_id || movie.MovieID}`}
                                alt={cleanTitle()}
                                className="w-full h-auto rounded-md"
                            />
                        </div>
                        
                        {/* Movie details */}
                        <div className="md:ml-6 flex-grow">
                            <h2 className="text-3xl font-bold">{cleanTitle()}</h2>
                            
                            <div className="flex flex-wrap items-center text-white text-opacity-80 mt-2">
                                {getYear() && <span className="mr-4">{getYear()}</span>}
                                
                                {/* Rating */}
                                {(movie.rating || movie.score || movie.HybridScore) && (
                                    <div className="flex items-center mr-4">
                                        <span className="text-yellow-400 mr-1">★</span>
                                        <span>{parseFloat(movie.rating || movie.score || movie.HybridScore).toFixed(1)}</span>
                                    </div>
                                )}
                                
                                {/* Context */}
                                {movie.time_of_day && movie.device_type && (
                                    <span className="text-netflix-red">
                                        For {movie.time_of_day} on {movie.device_type}
                                    </span>
                                )}
                            </div>
                            
                            {/* Genres */}
                            {movie.genres && (
                                <div className="flex flex-wrap mt-3">
                                    {movie.genres.map((genre, index) => (
                                        <span 
                                            key={index}
                                            className="bg-gray-800 text-white px-2 py-1 rounded mr-2 mb-2 text-sm"
                                        >
                                            {genre}
                                        </span>
                                    ))}
                                </div>
                            )}
                            
                            {/* Recommendation explanation */}
                            {getExplanationSection()}
                            
                            {/* Model contribution visualization */}
                            {getContributionChart()}
                        </div>
                    </div>
                    
                    {/* Similar movies section */}
                    <div className="mt-8 border-t border-gray-700 pt-6">
                        <h3 className="text-xl font-bold mb-4">More Like This</h3>
                        
                        {loading ? (
                            <div className="flex justify-center p-4">
                                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-netflix-red"></div>
                            </div>
                        ) : similarMovies.length > 0 ? (
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
                                {similarMovies.map((similarMovie) => (
                                    <div 
                                        key={similarMovie.movie_id}
                                        className="cursor-pointer"
                                    >
                                        <img 
                                            src={similarMovie.poster || `https://via.placeholder.com/150x225/141414/E50914?text=${similarMovie.movie_id}`}
                                            alt={similarMovie.title}
                                            className="w-full h-auto rounded-md hover:opacity-80 transition-opacity"
                                        />
                                        <h4 className="text-sm mt-1 line-clamp-1">{similarMovie.title}</h4>
                                        <div className="flex items-center text-xs text-gray-400">
                                            <span className="text-yellow-400 mr-1">★</span>
                                            <span>{parseFloat(similarMovie.similarity).toFixed(2)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-gray-400">No similar movies found.</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Export component to window object
window.MovieModal = MovieModal; 