// MovieCard.jsx - Individual movie card component

const MovieCard = ({ movie, onClick }) => {
    return (
        <div 
            className="movie-card relative w-full h-48 md:h-60 bg-netflix-dark-gray rounded overflow-hidden"
            onClick={() => onClick(movie)}
        >
            {/* Movie Poster */}
            <div className="absolute inset-0 bg-cover bg-center" 
                style={{ 
                    backgroundImage: `url(${movie.poster || 'https://via.placeholder.com/300x450?text=No+Image'})`,
                    filter: 'brightness(0.9)'
                }}>
            </div>
            
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent opacity-90"></div>
            
            {/* Content */}
            <div className="absolute bottom-0 left-0 right-0 p-3 text-white">
                <h3 className="font-semibold text-sm md:text-base line-clamp-2">{movie.title}</h3>
                
                {/* Movie info */}
                <div className="flex items-center mt-1">
                    <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                        <span className="text-xs ml-1">{movie.score?.toFixed(1) || 'N/A'}</span>
                    </div>
                    
                    {movie.genres && movie.genres.length > 0 && (
                        <div className="ml-2 text-xs text-gray-300 truncate">
                            {movie.genres.slice(0, 2).join(', ')}
                            {movie.genres.length > 2 && '...'}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}; 