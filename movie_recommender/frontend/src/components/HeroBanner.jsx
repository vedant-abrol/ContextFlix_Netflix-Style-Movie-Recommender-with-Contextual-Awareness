// HeroBanner.jsx - Featured movie hero banner like Netflix

const HeroBanner = ({ movie, onSelect }) => {
    // Default background image (used for placeholder or fallback)
    const defaultBg = "https://via.placeholder.com/1920x1080/141414/E50914?text=ContextFlix";
    
    // Clean title (remove year in parentheses)
    const cleanTitle = () => {
        if (!movie.title) return '';
        const yearMatch = movie.title.match(/\s*\(\d{4}\)$/);
        if (yearMatch) {
            return movie.title.replace(yearMatch[0], '');
        }
        return movie.title;
    };
    
    // Extract year from movie title or use provided year
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
    
    // Create explanation text based on available data
    const getExplanationText = () => {
        if (!movie.explanation) return '';
        
        let explanationText = '';
        
        if (movie.explanation.collaborative_filtering) {
            explanationText += movie.explanation.collaborative_filtering.explanation + '. ';
        }
        
        if (movie.explanation.content_based) {
            explanationText += movie.explanation.content_based.explanation + '. ';
        }
        
        if (movie.explanation.sequential) {
            explanationText += movie.explanation.sequential.explanation + '. ';
        }
        
        if (movie.explanation.context) {
            explanationText += movie.explanation.context.explanation + '. ';
        }
        
        return explanationText.trim();
    };
    
    // Get genre badges
    const getGenreBadges = () => {
        if (!movie.genres || !Array.isArray(movie.genres)) return null;
        
        return (
            <div className="flex flex-wrap mt-2 mb-4">
                {movie.genres.map((genre, index) => (
                    <span 
                        key={index}
                        className="text-sm bg-gray-800 bg-opacity-80 text-white px-3 py-1 rounded-full mr-2 mb-2"
                    >
                        {genre}
                    </span>
                ))}
            </div>
        );
    };
    
    // Get rating display
    const getRating = () => {
        const rating = movie.rating || movie.Rating || movie.PredictedRating || movie.score || movie.HybridScore;
        
        if (!rating) return null;
        
        return (
            <div className="flex items-center text-lg mr-4">
                <span className="text-yellow-400 mr-1">★</span>
                <span>{parseFloat(rating).toFixed(1)}</span>
            </div>
        );
    };
    
    // Background image for banner (use placeholder in real app we'd use movie.backdrop_url)
    const bannerStyle = {
        backgroundImage: `linear-gradient(to right, rgba(20, 20, 20, 0.9), rgba(20, 20, 20, 0.6), rgba(20, 20, 20, 0.4)), url(${defaultBg})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        height: '80vh'
    };
    
    return (
        <div className="hero-banner" style={bannerStyle}>
            <div className="flex flex-col justify-end h-full pb-20 px-8 md:px-16">
                {/* Movie title */}
                <h1 className="text-4xl md:text-6xl font-bold mb-2">{cleanTitle()}</h1>
                
                {/* Movie metadata */}
                <div className="flex items-center text-white text-opacity-80 mb-4">
                    {getYear() && <span className="mr-4">{getYear()}</span>}
                    {getRating()}
                    {movie.time_of_day && movie.device_type && (
                        <span className="text-netflix-red font-semibold">
                            Recommended for {movie.time_of_day} on {movie.device_type}
                        </span>
                    )}
                </div>
                
                {/* Genre badges */}
                {getGenreBadges()}
                
                {/* Explanation (recommendation reason) */}
                <p className="max-w-2xl text-lg mb-6">
                    {getExplanationText() || "Recommended for you based on your preferences and viewing history."}
                </p>
                
                {/* Action buttons */}
                <div className="flex space-x-4">
                    <button
                        className="netflix-button px-6 py-3 flex items-center"
                        onClick={() => onSelect(movie)}
                    >
                        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M8 5v14l11-7z" />
                        </svg>
                        Watch Now
                    </button>
                    
                    <button
                        className="bg-gray-700 bg-opacity-70 text-white px-6 py-3 rounded-md flex items-center hover:bg-gray-600"
                        onClick={() => onSelect(movie)}
                    >
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        More Info
                    </button>
                </div>
            </div>
            
            {/* Bottom gradient */}
            <div className="hero-gradient"></div>
        </div>
    );
};

// Export component to window object
window.HeroBanner = HeroBanner; 