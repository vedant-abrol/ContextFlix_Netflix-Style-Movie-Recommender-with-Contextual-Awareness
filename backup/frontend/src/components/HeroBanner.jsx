// HeroBanner.jsx - Hero banner for featured content

const HeroBanner = ({ movie, timeOfDay, onMovieSelect }) => {
    if (!movie) return null;
    
    // Get greeting based on time of day
    const getGreeting = () => {
        switch(timeOfDay) {
            case 'morning':
                return 'Good Morning';
            case 'afternoon':
                return 'Good Afternoon';
            case 'evening':
                return 'Good Evening';
            case 'night':
                return 'Good Night';
            default:
                return 'Welcome';
        }
    };
    
    return (
        <div className="hero-banner relative">
            {/* Background image */}
            <div 
                className="absolute inset-0 bg-cover bg-center"
                style={{ 
                    backgroundImage: `url(${movie.poster || 'https://via.placeholder.com/1200x800?text=Featured+Movie'})`,
                    filter: 'brightness(0.7)'
                }}
            />
            
            {/* Gradient overlay */}
            <div className="hero-gradient absolute inset-0"></div>
            
            {/* Content */}
            <div className="relative z-10 h-full flex flex-col justify-end p-6 md:p-12">
                <div className="max-w-3xl mb-16">
                    {/* Time-based greeting */}
                    <h3 className="text-xl text-gray-300 mb-2">{getGreeting()}</h3>
                    
                    {/* Movie title */}
                    <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">{movie.title}</h1>
                    
                    {/* Movie metadata */}
                    <div className="flex items-center mb-4">
                        <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                            <span className="ml-1 text-white">{movie.score?.toFixed(1) || 'N/A'}</span>
                        </div>
                        
                        {movie.genres && movie.genres.length > 0 && (
                            <div className="ml-4 text-gray-300">
                                {movie.genres.join(' • ')}
                            </div>
                        )}
                    </div>
                    
                    {/* Contextual recommendation explanation */}
                    <p className="text-gray-300 text-lg mb-6">
                        Top pick for your {timeOfDay} viewing
                    </p>
                    
                    {/* Action buttons */}
                    <div className="flex space-x-4">
                        <button 
                            onClick={() => onMovieSelect(movie)}
                            className="netflix-button px-6 py-3 flex items-center"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            More Info
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}; 