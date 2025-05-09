// ContextSelector.jsx - Component for selecting time of day and device type

const ContextSelector = ({ timeOfDay, deviceType, onContextChange }) => {
    const timeOptions = [
        { value: 'morning', label: 'Morning', icon: '🌅', description: '5 AM - 11:59 AM' },
        { value: 'afternoon', label: 'Afternoon', icon: '☀️', description: '12 PM - 4:59 PM' },
        { value: 'evening', label: 'Evening', icon: '🌆', description: '5 PM - 8:59 PM' },
        { value: 'night', label: 'Night', icon: '🌙', description: '9 PM - 4:59 AM' }
    ];
    
    const deviceOptions = [
        { value: 'mobile', label: 'Mobile', icon: '📱', description: 'Smartphones' },
        { value: 'tablet', label: 'Tablet', icon: '📲', description: 'iPads & Android tablets' },
        { value: 'desktop', label: 'Desktop', icon: '💻', description: 'Computers & laptops' },
        { value: 'TV', label: 'TV', icon: '📺', description: 'Smart TVs & consoles' }
    ];
    
    return (
        <div className="context-selector">
            <div className="mb-6">
                <h3 className="text-xl font-semibold mb-3">Time of Day</h3>
                <p className="text-netflix-light-gray mb-4">
                    Select when you're watching to get time-aware recommendations
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {timeOptions.map((option) => (
                        <button
                            key={option.value}
                            className={`flex flex-col items-center p-3 rounded-lg border ${
                                timeOfDay === option.value 
                                    ? 'border-netflix-red bg-netflix-red bg-opacity-20' 
                                    : 'border-gray-700 hover:border-gray-500'
                            }`}
                            onClick={() => onContextChange('time', option.value)}
                        >
                            <span className="text-2xl mb-1">{option.icon}</span>
                            <span className="font-medium">{option.label}</span>
                            <span className="text-xs text-netflix-light-gray mt-1">{option.description}</span>
                        </button>
                    ))}
                </div>
            </div>
            
            <div>
                <h3 className="text-xl font-semibold mb-3">Device Type</h3>
                <p className="text-netflix-light-gray mb-4">
                    Select your viewing device to personalize your experience
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {deviceOptions.map((option) => (
                        <button
                            key={option.value}
                            className={`flex flex-col items-center p-3 rounded-lg border ${
                                deviceType === option.value 
                                    ? 'border-netflix-red bg-netflix-red bg-opacity-20' 
                                    : 'border-gray-700 hover:border-gray-500'
                            }`}
                            onClick={() => onContextChange('device', option.value)}
                        >
                            <span className="text-2xl mb-1">{option.icon}</span>
                            <span className="font-medium">{option.label}</span>
                            <span className="text-xs text-netflix-light-gray mt-1">{option.description}</span>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

// Export component to window object
window.ContextSelector = ContextSelector; 