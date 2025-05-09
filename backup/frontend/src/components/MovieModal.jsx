// MovieModal.jsx - Detailed movie information in a modal

const MovieModal = ({ movie, onClose, currentUser, timeOfDay, deviceType, apiBaseUrl, apiAvailable }) => {
    const [similarMovies, setSimilarMovies] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const modalRef = React.useRef(null);
    
    // Fetch similar movies when the modal opens
    React.useEffect(() => {
        const fetchSimilarMovies = async () => {
            try {
                if (apiAvailable) {
                    const data = await ApiService.getSimilarMovies(movie.movie_id);
                    setSimilarMovies(data.similar_movies || []);
                } else {
                    // Mock similar movies data
                    const mockSimilar = Array.from({ length: 5 }, (_, i) => ({
                        movie_id: 100 + i,
                        title: `Similar Movie ${i + 1}`,
                        score: (Math.random() * 2) + 3,
                        poster: `https://via.placeholder.com/150x225.png?text=${100 + i}`,
                        genres: ['Genre 1', 'Genre 2']
                    }));
                    setSimilarMovies(mockSimilar);
                }
            } catch (error) {
                console.error('Error fetching similar movies:', error);
                setSimilarMovies([]);
            } finally {
                setLoading(false);
            }
        };
        
        fetchSimilarMovies();
    }, [movie.movie_id, apiAvailable, apiBaseUrl]);
    
    // Close when clicking outside the modal content
    React.useEffect(() => {
        const handleClickOutside = (event) => {
            if (modalRef.current && !modalRef.current.contains(event.target)) {
                onClose();
            }
        };
        
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [onClose]);
    
    return (
        <div className="movie-modal">
            <div 
                ref={modalRef}
                className="movie-modal-content"
            >
                {/* Header with poster */}
                <div className="relative h-64 sm:h-80 md:h-96 bg-netflix-dark-gray">
                    {/* Movie Poster/Banner */}
                    <div className="absolute inset-0 bg-cover bg-center" 
                        style={{ 
                            backgroundImage: `url(${movie.poster || 'https://via.placeholder.com/300x450?text=No+Image'})`,
                            filter: 'brightness(0.7)'
                        }}>
                    </div>
                    
                    {/* Gradient overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-netflix-dark-gray to-transparent"></div>
                    
                    {/* Close button */}
                    <button 
                        onClick={onClose}
                        className="absolute top-4 right-4 z-10 bg-netflix-black bg-opacity-70 rounded-full p-2"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                    
                    {/* Movie title and basics */}
                    <div className="absolute bottom-0 left-0 right-0 p-6 text-white">
                        <h2 className="text-2xl md:text-3xl font-bold">{movie.title}</h2>
                        
                        <div className="flex items-center mt-2 text-sm">
                            <div className="flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                </svg>
                                <span className="ml-1">{movie.score?.toFixed(1) || 'N/A'}</span>
                            </div>
                            
                            {movie.genres && movie.genres.length > 0 && (
                                <div className="ml-4 text-gray-300">
                                    {movie.genres.join(', ')}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                
                {/* Movie content */}
                <div className="p-6">
                    {/* Recommendation contextual explanation */}
                    <div className="mb-6 p-4 bg-netflix-black rounded-lg text-sm">
                        <h3 className="text-netflix-red font-semibold mb-2">Why this is recommended for you</h3>
                        <p>
                            This movie matches your preferences for {movie.genres && movie.genres.slice(0, 2).join(' and ')} 
                            content. Based on your viewing history, you tend to watch similar content
                            during {timeOfDay} on your {deviceType} device.
                        </p>
                    </div>
                    
                    {/* Similar movies section */}
                    <div className="mt-8">
                        <h3 className="text-xl font-semibold mb-4">More Like This</h3>
                        
                        {loading ? (
                            <div className="flex justify-center p-4">
                                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-netflix-red"></div>
                            </div>
                        ) : similarMovies.length > 0 ? (
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                                {similarMovies.map(movie => (
                                    <div key={movie.movie_id} className="aspect-[2/3] relative rounded overflow-hidden">
                                        <div 
                                            className="absolute inset-0 bg-cover bg-center"
                                            style={{ backgroundImage: `url(${movie.poster || 'https://via.placeholder.com/300x450?text=No+Image'})` }}
                                        ></div>
                                        <div className="absolute inset-0 bg-black bg-opacity-40 flex items-end">
                                            <div className="p-2 text-xs text-white">{movie.title}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-gray-400 text-center py-4">No similar movies found</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}; 