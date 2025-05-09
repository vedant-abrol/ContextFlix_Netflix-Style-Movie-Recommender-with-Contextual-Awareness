// Navbar.jsx - Top navigation component with user selection and context options

const Navbar = ({ currentUser, users, onUserSelect, timeOfDay, deviceType, onContextChange }) => {
    const [isUserMenuOpen, setIsUserMenuOpen] = React.useState(false);
    const [isContextMenuOpen, setIsContextMenuOpen] = React.useState(false);
    
    const toggleUserMenu = () => {
        setIsUserMenuOpen(!isUserMenuOpen);
        if (isContextMenuOpen) setIsContextMenuOpen(false);
    };
    
    const toggleContextMenu = () => {
        setIsContextMenuOpen(!isContextMenuOpen);
        if (isUserMenuOpen) setIsUserMenuOpen(false);
    };
    
    return (
        <nav className="bg-netflix-black py-3 px-4 md:px-8 flex items-center justify-between shadow-md">
            {/* Logo */}
            <div className="flex items-center">
                <h1 className="text-netflix-red text-2xl md:text-3xl font-bold">
                    ContextFlix
                </h1>
            </div>
            
            {/* Nav Links (desktop) */}
            <div className="hidden md:flex space-x-6">
                <a href="/" className="text-netflix-white hover:text-netflix-red transition-colors">
                    Home
                </a>
                <a href="/search" className="text-netflix-white hover:text-netflix-red transition-colors">
                    Search
                </a>
                {currentUser && (
                    <a href={`/user/${currentUser.id}/history`} className="text-netflix-white hover:text-netflix-red transition-colors">
                        History
                    </a>
                )}
            </div>
            
            {/* Context and User Selection */}
            <div className="flex items-center space-x-4">
                {/* Context Selector */}
                <div className="relative">
                    <button 
                        onClick={toggleContextMenu}
                        className="flex items-center space-x-1 text-netflix-white hover:text-netflix-red"
                    >
                        <span>Context: {timeOfDay} / {deviceType}</span>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path fillRule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                        </svg>
                    </button>
                    
                    {isContextMenuOpen && (
                        <div className="absolute right-0 mt-2 w-60 bg-netflix-dark-gray rounded shadow-lg z-20">
                            <div className="p-3 border-b border-gray-700">
                                <h3 className="text-netflix-white font-semibold">Time of Day</h3>
                                <div className="mt-2 space-y-1">
                                    {['morning', 'afternoon', 'evening', 'night'].map(time => (
                                        <button
                                            key={time}
                                            onClick={() => {
                                                onContextChange('time', time);
                                                setIsContextMenuOpen(false);
                                            }}
                                            className={`block w-full text-left px-2 py-1 rounded ${
                                                timeOfDay === time 
                                                    ? 'bg-netflix-red text-white' 
                                                    : 'text-white hover:bg-gray-700'
                                            }`}
                                        >
                                            {time.charAt(0).toUpperCase() + time.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            
                            <div className="p-3">
                                <h3 className="text-netflix-white font-semibold">Device Type</h3>
                                <div className="mt-2 space-y-1">
                                    {['mobile', 'tablet', 'desktop'].map(device => (
                                        <button
                                            key={device}
                                            onClick={() => {
                                                onContextChange('device', device);
                                                setIsContextMenuOpen(false);
                                            }}
                                            className={`block w-full text-left px-2 py-1 rounded ${
                                                deviceType === device 
                                                    ? 'bg-netflix-red text-white' 
                                                    : 'text-white hover:bg-gray-700'
                                            }`}
                                        >
                                            {device.charAt(0).toUpperCase() + device.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                
                {/* User Selector */}
                <div className="relative">
                    <button 
                        onClick={toggleUserMenu}
                        className="flex items-center space-x-2 text-netflix-white hover:text-netflix-red"
                    >
                        <div className="w-8 h-8 bg-netflix-red rounded-full flex items-center justify-center">
                            {currentUser ? (
                                <span className="font-bold">{currentUser.id}</span>
                            ) : (
                                <span>?</span>
                            )}
                        </div>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path fillRule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                        </svg>
                    </button>
                    
                    {isUserMenuOpen && (
                        <div className="absolute right-0 mt-2 w-48 bg-netflix-dark-gray rounded shadow-lg z-20">
                            <div className="py-1">
                                {users.map(user => (
                                    <button
                                        key={user.id}
                                        onClick={() => {
                                            onUserSelect(user);
                                            setIsUserMenuOpen(false);
                                        }}
                                        className={`block w-full text-left px-4 py-2 ${
                                            currentUser?.id === user.id 
                                                ? 'bg-netflix-red text-white' 
                                                : 'text-white hover:bg-gray-700'
                                        }`}
                                    >
                                        User {user.id} {user.gender && `(${user.gender})`}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
}; 