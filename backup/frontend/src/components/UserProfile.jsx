// UserProfile.jsx - User profile information and preferences

const UserProfile = ({ user, apiBaseUrl, apiAvailable }) => {
    const [preferences, setPreferences] = React.useState(null);
    const [loading, setLoading] = React.useState(true);
    
    React.useEffect(() => {
        const fetchPreferences = async () => {
            try {
                if (apiAvailable && user) {
                    const data = await ApiService.getUserPreferences(user.id);
                    setPreferences(data);
                } else {
                    // Mock preferences
                    setPreferences({
                        favorite_genres: ['Action', 'Comedy', 'Drama'],
                        avg_rating: user?.avg_rating || 3.8,
                        active_times: {
                            morning: 15,
                            afternoon: 20,
                            evening: 45,
                            night: 20
                        },
                        device_usage: {
                            mobile: 30,
                            tablet: 15,
                            desktop: 55
                        }
                    });
                }
            } catch (error) {
                console.error('Error fetching user preferences:', error);
            } finally {
                setLoading(false);
            }
        };
        
        fetchPreferences();
    }, [user, apiAvailable, apiBaseUrl]);
    
    if (!user) {
        return (
            <div className="bg-netflix-dark-gray rounded-lg p-4 text-center text-gray-400">
                No user selected
            </div>
        );
    }
    
    if (loading) {
        return (
            <div className="bg-netflix-dark-gray rounded-lg p-4 flex justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-netflix-red"></div>
            </div>
        );
    }
    
    // Helper function to get the most active time
    const getMostActiveTime = () => {
        if (!preferences?.active_times) return 'evening';
        
        let maxTime = 'evening';
        let maxValue = 0;
        
        Object.entries(preferences.active_times).forEach(([time, value]) => {
            if (value > maxValue) {
                maxValue = value;
                maxTime = time;
            }
        });
        
        return maxTime;
    };
    
    // Helper function to get the most used device
    const getMostUsedDevice = () => {
        if (!preferences?.device_usage) return 'desktop';
        
        let maxDevice = 'desktop';
        let maxValue = 0;
        
        Object.entries(preferences.device_usage).forEach(([device, value]) => {
            if (value > maxValue) {
                maxValue = value;
                maxDevice = device;
            }
        });
        
        return maxDevice;
    };
    
    return (
        <div className="bg-netflix-dark-gray rounded-lg overflow-hidden">
            {/* User header */}
            <div className="bg-gradient-to-r from-netflix-red to-red-800 p-4">
                <div className="flex items-center">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center text-netflix-red font-bold text-2xl">
                        {user.id}
                    </div>
                    <div className="ml-4">
                        <h3 className="text-white text-xl font-semibold">User {user.id}</h3>
                        <p className="text-gray-200">
                            {user.gender && `${user.gender === 'M' ? 'Male' : 'Female'} • `}
                            {user.rating_count || 0} ratings
                        </p>
                    </div>
                </div>
            </div>
            
            {/* User preferences */}
            <div className="p-4">
                {/* Favorite genres */}
                <div className="mb-4">
                    <h4 className="text-gray-400 text-sm mb-2">Favorite Genres</h4>
                    <div className="flex flex-wrap gap-2">
                        {preferences?.favorite_genres?.map(genre => (
                            <span 
                                key={genre} 
                                className="bg-netflix-black px-2 py-1 rounded text-sm text-white"
                            >
                                {genre}
                            </span>
                        ))}
                    </div>
                </div>
                
                {/* Average rating */}
                <div className="mb-4">
                    <h4 className="text-gray-400 text-sm mb-2">Average Rating</h4>
                    <div className="flex items-center">
                        <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                            <span className="ml-1 text-white">{preferences?.avg_rating?.toFixed(1) || user.avg_rating?.toFixed(1) || 'N/A'}</span>
                        </div>
                        <span className="ml-1 text-gray-400">/ 5.0</span>
                    </div>
                </div>
                
                {/* Viewing patterns */}
                <div className="mb-4">
                    <h4 className="text-gray-400 text-sm mb-2">Viewing Patterns</h4>
                    <p className="text-white">
                        Most active during <span className="text-netflix-red">{getMostActiveTime()}</span> on <span className="text-netflix-red">{getMostUsedDevice()}</span>
                    </p>
                </div>
            </div>
        </div>
    );
}; 