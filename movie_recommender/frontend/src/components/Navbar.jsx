// Navbar.jsx - Navigation bar component with user and context selection

const Navbar = ({ currentUser, users, onUserSelect, timeOfDay, deviceType, onContextChange }) => {
    const [showUserMenu, setShowUserMenu] = React.useState(false);
    const [showContextMenu, setShowContextMenu] = React.useState(false);
    
    // Toggle user dropdown menu
    const toggleUserMenu = () => {
        setShowUserMenu(!showUserMenu);
        if (showContextMenu) setShowContextMenu(false);
    };
    
    // Toggle context dropdown menu
    const toggleContextMenu = () => {
        setShowContextMenu(!showContextMenu);
        if (showUserMenu) setShowUserMenu(false);
    };
    
    return (
        <nav className="flex items-center justify-between p-4 bg-black bg-opacity-90 fixed top-0 left-0 right-0 z-50">
            {/* Logo */}
            <div className="flex items-center">
                <a href="/" className="text-netflix-red text-3xl font-bold">
                    ContextFlix
                </a>
                
                {/* Navigation links */}
                {currentUser && (
                    <div className="ml-8 hidden md:flex space-x-6">
                        <a href="/" className="text-white hover:text-gray-300">Home</a>
                        <a href={`/user/${currentUser.id}/history`} className="text-white hover:text-gray-300">My History</a>
                    </div>
                )}
            </div>
            
            {/* Right side controls */}
            <div className="flex items-center space-x-4">
                {/* Context selector */}
                {currentUser && (
                    <div className="relative">
                        <button 
                            className="flex items-center px-3 py-2 bg-black border border-gray-700 rounded-md"
                            onClick={toggleContextMenu}
                        >
                            <span className="text-white mr-2">
                                {timeOfDay}, {deviceType}
                            </span>
                            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                            </svg>
                        </button>
                        
                        {/* Context dropdown menu */}
                        {showContextMenu && (
                            <div className="absolute right-0 mt-2 w-64 bg-black border border-gray-700 rounded-md shadow-lg overflow-hidden z-10">
                                <div className="p-4">
                                    <h3 className="text-white text-lg font-semibold mb-2">Time of Day</h3>
                                    <div className="grid grid-cols-2 gap-2 mb-4">
                                        {['morning', 'afternoon', 'evening', 'night'].map(time => (
                                            <button
                                                key={time}
                                                className={`px-3 py-2 rounded ${timeOfDay === time ? 'bg-netflix-red' : 'bg-gray-800 hover:bg-gray-700'}`}
                                                onClick={() => {
                                                    onContextChange('time', time);
                                                    setShowContextMenu(false);
                                                }}
                                            >
                                                {time}
                                            </button>
                                        ))}
                                    </div>
                                    
                                    <h3 className="text-white text-lg font-semibold mb-2">Device Type</h3>
                                    <div className="grid grid-cols-2 gap-2">
                                        {['mobile', 'tablet', 'desktop', 'TV'].map(device => (
                                            <button
                                                key={device}
                                                className={`px-3 py-2 rounded ${deviceType === device ? 'bg-netflix-red' : 'bg-gray-800 hover:bg-gray-700'}`}
                                                onClick={() => {
                                                    onContextChange('device', device);
                                                    setShowContextMenu(false);
                                                }}
                                            >
                                                {device}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
                
                {/* User profile */}
                <div className="relative">
                    <button 
                        className="flex items-center"
                        onClick={toggleUserMenu}
                    >
                        <div className="w-8 h-8 rounded-md bg-netflix-red flex items-center justify-center text-white">
                            {currentUser ? currentUser.id : '?'}
                        </div>
                        <svg className="w-4 h-4 text-white ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                    </button>
                    
                    {/* User dropdown menu */}
                    {showUserMenu && (
                        <div className="absolute right-0 mt-2 w-48 bg-black border border-gray-700 rounded-md shadow-lg overflow-hidden z-10">
                            <div className="py-1">
                                {users.map(user => (
                                    <button
                                        key={user.id}
                                        className={`w-full px-4 py-2 text-left ${currentUser && currentUser.id === user.id ? 'bg-gray-800' : 'hover:bg-gray-800'}`}
                                        onClick={() => {
                                            onUserSelect(user);
                                            setShowUserMenu(false);
                                        }}
                                    >
                                        <div className="flex items-center">
                                            <div className="w-6 h-6 rounded-md bg-netflix-red flex items-center justify-center text-white mr-2">
                                                {user.id}
                                            </div>
                                            <span>
                                                User {user.id}
                                                {user.gender && ` (${user.gender})`}
                                            </span>
                                        </div>
                                    </button>
                                ))}
                                <div className="border-t border-gray-700 my-1"></div>
                                <a
                                    href="/select-user"
                                    className="block px-4 py-2 hover:bg-gray-800"
                                >
                                    Select Profile
                                </a>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
}; 

// Export component to window object
window.Navbar = Navbar; 