// GenreList.jsx - Component for displaying a list of movie genres

const GenreList = ({ genres, onSelectGenre, selectedGenre }) => {
    if (!genres || genres.length === 0) {
        return (
            <div className="text-center py-4">
                <p className="text-netflix-light-gray">No genres available.</p>
            </div>
        );
    }
    
    return (
        <div className="genre-list">
            <h3 className="text-xl font-semibold mb-3">Genres</h3>
            
            <div className="flex flex-wrap">
                <button
                    className={`px-3 py-1 rounded-full mr-2 mb-2 text-sm ${
                        !selectedGenre 
                            ? 'bg-netflix-red text-white' 
                            : 'bg-gray-800 text-white hover:bg-gray-700'
                    }`}
                    onClick={() => onSelectGenre(null)}
                >
                    All
                </button>
                
                {genres.map((genre) => (
                    <button
                        key={genre.name}
                        className={`px-3 py-1 rounded-full mr-2 mb-2 text-sm ${
                            selectedGenre === genre.name 
                                ? 'bg-netflix-red text-white' 
                                : 'bg-gray-800 text-white hover:bg-gray-700'
                        }`}
                        onClick={() => onSelectGenre(genre.name)}
                    >
                        {genre.name} {genre.count && <span className="text-xs">({genre.count})</span>}
                    </button>
                ))}
            </div>
        </div>
    );
};

// Export component to window object
window.GenreList = GenreList; 