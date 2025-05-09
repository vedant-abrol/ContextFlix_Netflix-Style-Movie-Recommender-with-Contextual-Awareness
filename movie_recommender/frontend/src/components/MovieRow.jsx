// MovieRow.jsx - Horizontal scrolling row of movies like Netflix

const MovieRow = ({ title, movies, onMovieSelect }) => {
    const scrollContainerRef = React.useRef(null);
    
    // Handle left and right scrolling
    const scroll = (direction) => {
        const container = scrollContainerRef.current;
        if (!container) return;
        
        const scrollAmount = container.clientWidth * 0.8; // Scroll 80% of container width
        const currentScroll = container.scrollLeft;
        
        container.scrollTo({
            left: direction === 'left' 
                ? currentScroll - scrollAmount 
                : currentScroll + scrollAmount,
            behavior: 'smooth'
        });
    };
    
    // Don't render if no movies
    if (!movies || movies.length === 0) {
        return null;
    }
    
    return (
        <div className="carousel-row mb-10">
            {/* Row title */}
            <h2 className="text-xl font-bold mb-4">{title}</h2>
            
            {/* Scroll container and controls */}
            <div className="relative group">
                {/* Left scroll button */}
                <button 
                    className="absolute left-0 top-0 bottom-0 z-10 w-12 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => scroll('left')}
                >
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
                    </svg>
                </button>
                
                {/* Movie cards container */}
                <div 
                    ref={scrollContainerRef}
                    className="flex space-x-4 overflow-x-scroll pb-4 no-scrollbar"
                >
                    {movies.map((movie, index) => (
                        <div 
                            key={movie.movie_id || movie.MovieID || index} 
                            className="flex-shrink-0 w-32 sm:w-40 md:w-48"
                        >
                            <MovieCard 
                                movie={movie} 
                                onSelect={onMovieSelect}
                            />
                        </div>
                    ))}
                </div>
                
                {/* Right scroll button */}
                <button 
                    className="absolute right-0 top-0 bottom-0 z-10 w-12 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => scroll('right')}
                >
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                    </svg>
                </button>
            </div>
        </div>
    );
};

// Export component to window object
window.MovieRow = MovieRow; 