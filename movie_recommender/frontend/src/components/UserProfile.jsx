// UserProfile.jsx - Component for displaying user profile information

const UserProfile = ({ user }) => {
    if (!user) {
        return null;
    }
    
    return (
        <div className="user-profile p-4">
            <h2 className="text-2xl font-bold mb-4">User Profile</h2>
            
            <div className="bg-netflix-dark-gray rounded-md p-4">
                <div className="flex items-center mb-4">
                    <div className="w-16 h-16 rounded-md bg-netflix-red flex items-center justify-center text-2xl text-white mr-4">
                        {user.id}
                    </div>
                    <div>
                        <h3 className="text-xl font-semibold">User {user.id}</h3>
                        {user.gender && <p className="text-netflix-light-gray">Gender: {user.gender}</p>}
                        {user.age && <p className="text-netflix-light-gray">Age: {user.age}</p>}
                        {user.occupation && <p className="text-netflix-light-gray">Occupation: {user.occupation}</p>}
                    </div>
                </div>
                
                <div className="mt-4">
                    <p className="text-sm text-netflix-light-gray">Activity:</p>
                    <p>Total Ratings: {user.rating_count || 0}</p>
                    <p>Average Rating: {user.avg_rating ? user.avg_rating.toFixed(1) : 'N/A'}</p>
                </div>
            </div>
        </div>
    );
};

// Export component to window object
window.UserProfile = UserProfile; 