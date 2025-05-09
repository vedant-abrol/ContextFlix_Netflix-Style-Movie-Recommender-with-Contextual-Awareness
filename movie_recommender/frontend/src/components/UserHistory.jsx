// UserHistory.jsx - Component for displaying user viewing history

const UserHistory = ({ history }) => {
    if (!history || history.length === 0) {
        return (
            <div className="text-center py-10">
                <p className="text-lg text-netflix-light-gray">No viewing history available.</p>
            </div>
        );
    }
    
    return (
        <div className="user-history-container">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {history.map((item, index) => (
                    <div key={index} className="bg-netflix-dark-gray rounded-md p-4 shadow">
                        <div className="flex">
                            <img 
                                src={item.poster || `https://via.placeholder.com/150x225.png?text=${item.movie_id}`} 
                                alt={item.title}
                                className="w-20 h-30 object-cover rounded mr-4"
                            />
                            <div>
                                <h3 className="text-lg font-semibold line-clamp-2">{item.title}</h3>
                                
                                <div className="flex items-center mt-2">
                                    <span className="text-yellow-400 mr-1">★</span>
                                    <span>{item.rating.toFixed(1)}</span>
                                </div>
                                
                                {item.time_of_day && item.device_type && (
                                    <div className="text-netflix-light-gray text-sm mt-2">
                                        Watched in the {item.time_of_day} on {item.device_type}
                                    </div>
                                )}
                                
                                <div className="text-xs text-netflix-light-gray mt-2">
                                    {new Date(item.timestamp).toLocaleDateString()}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Export component to window object
window.UserHistory = UserHistory; 