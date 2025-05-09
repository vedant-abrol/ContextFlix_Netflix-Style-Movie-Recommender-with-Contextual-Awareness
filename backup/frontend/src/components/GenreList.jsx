// GenreList.jsx - Displays a list of genres and allows filtering by them

const GenreList = ({ apiBaseUrl, apiAvailable, onGenreSelect }) => {
    const [genres, setGenres] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);

    React.useEffect(() => {
        const fetchGenres = async () => {
            setLoading(true);
            setError(null);
            try {
                if (apiAvailable) {
                    const fetchedGenres = await ApiService.getGenres();
                    setGenres(fetchedGenres || []);
                } else {
                    // Mock genres
                    setGenres(['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 'Romance', 'Horror', 'Animation', 'Documentary', 'Fantasy']);
                }
            } catch (err) {
                console.error("Error fetching genres:", err);
                setError("Failed to load genres.");
            } finally {
                setLoading(false);
            }
        };

        fetchGenres();
    }, [apiAvailable, apiBaseUrl]);

    if (loading) {
        return (
            <div className="flex justify-center p-4">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }

    if (error) {
        return <div className="p-4 text-center text-netflix-red">{error}</div>;
    }

    return (
        <div className="bg-netflix-dark-gray p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-3">Browse by Genre</h3>
            {genres.length === 0 ? (
                <p className="text-gray-400">No genres available.</p>
            ) : (
                <div className="flex flex-wrap gap-2">
                    {genres.map(genre => (
                        <button 
                            key={genre}
                            onClick={() => onGenreSelect && onGenreSelect(genre)}
                            className="bg-netflix-black text-white px-3 py-1 rounded-full text-sm hover:bg-netflix-red transition-colors"
                        >
                            {genre}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}; 