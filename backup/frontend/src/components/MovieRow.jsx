// MovieRow.jsx - Horizontal row of movies with title

const MovieRow = ({ title, movies, onMovieSelect }) => {
    const rowRef = React.useRef(null);
    
    const scroll = (direction) => {
        const { current } = rowRef;
        if (current) {
            const scrollAmount = direction === 'left' 
                ? -current.offsetWidth * 0.75 
                : current.offsetWidth * 0.75;
            
            current.scrollBy({
                left: scrollAmount,
                behavior: 'smooth'
            });
        }
    };
    
    if (!movies || movies.length === 0) {
        return null;
    }
    
    return (
        <div className="carousel-row mb-8">
            {/* Row title */}
            <h2 className="text-xl md:text-2xl font-bold text-white mb-4">{title}</h2>
            
            {/* Carousel container */}
            <div className="relative group">
                {/* Scroll buttons */}
                {movies.length > 4 && (
                    <>
                        <button 
                            onClick={() => scroll('left')}
                            className="absolute left-0 top-0 bottom-0 z-10 w-12 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-r from-netflix-black to-transparent"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                        </button>
                        
                        <button 
                            onClick={() => scroll('right')}
                            className="absolute right-0 top-0 bottom-0 z-10 w-12 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-l from-netflix-black to-transparent"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                        </button>
                    </>
                )}
                
                {/* Movie cards container */}
                <div 
                    ref={rowRef}
                    className="flex space-x-4 overflow-x-auto scrollbar-hide py-2"
                    style={{ 
                        scrollbarWidth: 'none',
                        msOverflowStyle: 'none'
                    }}
                >
                    {movies.map(movie => (
                        <div key={movie.movie_id} className="flex-none w-[200px] md:w-[240px]">
                            <MovieCard movie={movie} onClick={onMovieSelect} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}; 