// UserHistory.jsx - Displays user's viewing history

const UserHistoryPage = ({ currentUser, apiBaseUrl, apiAvailable, handleMovieSelect }) => {
    const [history, setHistory] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);

    React.useEffect(() => {
        const fetchHistory = async () => {
            if (!currentUser) return;
            setLoading(true);
            setError(null);
            try {
                if (apiAvailable) {
                    const data = await ApiService.getUserHistory(currentUser.id, 50); // Fetch last 50 items
                    setHistory(data.history || []);
                } else {
                    // Mock history data
                    const mockHistory = Array.from({ length: 10 }, (_, i) => ({
                        movie_id: 200 + i,
                        title: `Watched Movie ${i + 1}`,
                        watched_at: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
                        rating: (Math.random() * 2) + 3,
                        poster: `https://via.placeholder.com/100x150.png?text=Hist${i+1}`,
                        genres: ['Action', 'Thriller']
                    }));
                    setHistory(mockHistory);
                }
            } catch (err) {
                console.error("Error fetching user history:", err);
                setError("Failed to load viewing history.");
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, [currentUser, apiAvailable, apiBaseUrl]);

    if (!currentUser) {
        return <div className="p-4 text-center text-gray-400">Please select a user to see their history.</div>;
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }

    if (error) {
        return <div className="p-4 text-center text-netflix-red">{error}</div>;
    }

    return (
        <div className="container mx-auto p-4 md:p-8">
            <h1 className="text-2xl md:text-3xl font-bold text-white mb-6">Viewing History for User {currentUser.id}</h1>
            
            {history.length === 0 ? (
                <p className="text-gray-400">No viewing history found for this user.</p>
            ) : (
                <div className="space-y-4">
                    {history.map(item => (
                        <div 
                            key={item.movie_id + item.watched_at}
                            className="bg-netflix-dark-gray p-4 rounded-lg flex items-center space-x-4 hover:bg-gray-700 transition-colors cursor-pointer"
                            onClick={() => handleMovieSelect && handleMovieSelect({ movie_id: item.movie_id, title: item.title, poster: item.poster, genres: item.genres, score: item.rating})}
                        >
                            <img src={item.poster || 'https://via.placeholder.com/80x120.png?text=No+Image'} alt={item.title} className="w-20 h-30 rounded"/>
                            <div className="flex-grow">
                                <h3 className="text-lg font-semibold text-white">{item.title}</h3>
                                <p className="text-sm text-gray-400">
                                    Watched on: {new Date(item.watched_at).toLocaleDateString()}
                                </p>
                                {item.rating && (
                                    <p className="text-sm text-yellow-400">Your Rating: {item.rating.toFixed(1)}</p>
                                )}
                            </div>
                            <button className="netflix-button text-sm">View Details</button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}; 