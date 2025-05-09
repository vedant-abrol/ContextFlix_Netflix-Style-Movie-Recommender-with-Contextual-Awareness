// MovieCard.jsx - Card component for displaying movie thumbnails

const MovieCard = ({ movie, onSelect, className = "" }) => {
    // Extract year from title if available
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
    
    // Clean title (remove year in parentheses)
    const cleanTitle = () => {
        if (!movie.title) return '';
        const yearMatch = movie.title.match(/\s*\(\d{4}\)$/);
        if (yearMatch) {
            return movie.title.replace(yearMatch[0], '');
        }
        return movie.title;
    };
    
    // Get movie poster image
    const getPoster = () => {
        return movie.poster || `https://via.placeholder.com/150x225.png?text=${movie.movie_id || movie.MovieID}`;
    };
    
    // Display genres as badges
    const renderGenres = () => {
        if (!movie.genres || !Array.isArray(movie.genres) || movie.genres.length === 0) {
            return null;
        }
        
        // Only show first 2 genres to avoid cluttering
        const displayGenres = movie.genres.slice(0, 2);
        
        return (
            <div className="flex flex-wrap mt-1">
                {displayGenres.map((genre, index) => (
                    <span 
                        key={index}
                        className="text-xs bg-gray-800 text-gray-300 px-2 py-1 rounded mr-1 mb-1"
                    >
                        {genre}
                    </span>
                ))}
            </div>
        );
    };
    
    // Display rating if available
    const renderRating = () => {
        const rating = movie.rating || movie.Rating || movie.PredictedRating || movie.score || movie.HybridScore;
        
        if (!rating) return null;
        
        return (
            <div className="flex items-center mt-1">
                <span className="text-yellow-400 mr-1">★</span>
                <span className="text-xs">{parseFloat(rating).toFixed(1)}</span>
            </div>
        );
    };
    
    // Render explanation hint if available
    const renderExplanationHint = () => {
        if (!movie.explanation) return null;
        
        let hint = '';
        
        // Check what type of explanation we have
        if (movie.explanation.collaborative_filtering) {
            hint = movie.explanation.collaborative_filtering.explanation;
        } else if (movie.explanation.content_based) {
            hint = movie.explanation.content_based.explanation;
        } else if (movie.explanation.sequential) {
            hint = movie.explanation.sequential.explanation;
        } else if (movie.contributions) {
            // If we have model contributions, use the highest one
            const contribs = movie.contributions;
            const maxContrib = Math.max(
                contribs.collaborative || 0,
                contribs.content_based || 0,
                contribs.sequential || 0
            );
            
            if (maxContrib === contribs.collaborative) {
                hint = "Based on similar users' ratings";
            } else if (maxContrib === contribs.content_based) {
                hint = "Matches your genre preferences";
            } else if (maxContrib === contribs.sequential) {
                hint = "Based on your viewing history";
            }
        }
        
        if (!hint) return null;
        
        return (
            <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                {hint}
            </div>
        );
    };
    
    return (
        <div 
            className={`movie-card relative group ${className}`}
            onClick={() => onSelect(movie)}
        >
            {/* Movie poster */}
            <img 
                src={getPoster()} 
                alt={cleanTitle()}
                className="w-full h-48 object-cover rounded-md"
                loading="lazy"
            />
            
            {/* Hover overlay with details */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <h3 className="text-sm font-semibold line-clamp-1">{cleanTitle()}</h3>
                
                <div className="flex items-center text-xs text-gray-300">
                    {getYear() && <span className="mr-1">{getYear()}</span>}
                    {renderRating()}
                </div>
                
                {renderGenres()}
                {renderExplanationHint()}
            </div>
            
            {/* Play button overlay on hover */}
            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="w-12 h-12 rounded-full bg-white bg-opacity-30 flex items-center justify-center border-2 border-white">
                    <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                    </svg>
                </div>
            </div>
        </div>
    );
};

// Export component to window object
window.MovieCard = MovieCard; 