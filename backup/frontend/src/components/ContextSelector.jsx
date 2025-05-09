// ContextSelector.jsx - Component for selecting context (time of day, device type)

const ContextSelector = ({ timeOfDay, deviceType, onContextChange }) => {
    return (
        <div className="bg-netflix-dark-gray rounded-lg p-4">
            <h3 className="text-white text-lg font-semibold mb-4">Context Settings</h3>
            
            {/* Time of day selector */}
            <div className="mb-6">
                <h4 className="text-gray-400 text-sm mb-2">Time of Day</h4>
                <div className="grid grid-cols-2 gap-2">
                    {['morning', 'afternoon', 'evening', 'night'].map(time => (
                        <button
                            key={time}
                            onClick={() => onContextChange('time', time)}
                            className={`py-2 px-4 rounded text-sm font-medium transition-colors ${
                                timeOfDay === time
                                    ? 'bg-netflix-red text-white'
                                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                            }`}
                        >
                            {time.charAt(0).toUpperCase() + time.slice(1)}
                        </button>
                    ))}
                </div>
            </div>
            
            {/* Device type selector */}
            <div>
                <h4 className="text-gray-400 text-sm mb-2">Device Type</h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {['mobile', 'tablet', 'desktop'].map(device => (
                        <button
                            key={device}
                            onClick={() => onContextChange('device', device)}
                            className={`py-2 px-4 rounded text-sm font-medium transition-colors ${
                                deviceType === device
                                    ? 'bg-netflix-red text-white'
                                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                            }`}
                        >
                            {device.charAt(0).toUpperCase() + device.slice(1)}
                        </button>
                    ))}
                </div>
            </div>
            
            {/* Info note */}
            <div className="mt-4 text-xs text-gray-400">
                <p>
                    Changing these settings will personalize recommendations based on your selected context.
                </p>
            </div>
        </div>
    );
}; 