// Explanation.jsx - Component for displaying recommendation explanations

const Explanation = ({ explanation }) => {
    if (!explanation) {
        return null;
    }
    
    // Extract explanation data
    const getCollaborativeExplanation = () => {
        if (explanation.collaborative_filtering) {
            return (
                <div className="explanation-section mb-3">
                    <h4 className="text-md font-semibold mb-1">Similar Users</h4>
                    <p className="text-sm text-netflix-light-gray">{explanation.collaborative_filtering.explanation}</p>
                </div>
            );
        }
        return null;
    };
    
    const getContentBasedExplanation = () => {
        if (explanation.content_based) {
            return (
                <div className="explanation-section mb-3">
                    <h4 className="text-md font-semibold mb-1">Genre Preferences</h4>
                    <p className="text-sm text-netflix-light-gray">{explanation.content_based.explanation}</p>
                    
                    {explanation.content_based.genres && explanation.content_based.genres.length > 0 && (
                        <div className="flex flex-wrap mt-2">
                            {explanation.content_based.genres.map((genre, index) => (
                                <span 
                                    key={index} 
                                    className="bg-gray-800 text-white px-2 py-1 rounded mr-2 mb-2 text-xs"
                                >
                                    {genre.name} ({(genre.importance * 100).toFixed(0)}%)
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            );
        }
        return null;
    };
    
    const getSequentialExplanation = () => {
        if (explanation.sequential) {
            return (
                <div className="explanation-section mb-3">
                    <h4 className="text-md font-semibold mb-1">Viewing History</h4>
                    <p className="text-sm text-netflix-light-gray">{explanation.sequential.explanation}</p>
                </div>
            );
        }
        return null;
    };
    
    const getContextExplanation = () => {
        if (explanation.context) {
            return (
                <div className="explanation-section mb-3">
                    <h4 className="text-md font-semibold mb-1">Viewing Context</h4>
                    <p className="text-sm text-netflix-light-gray">{explanation.context.explanation}</p>
                    <div className="flex mt-2">
                        <span className="bg-netflix-red text-white px-2 py-1 rounded mr-2 text-xs">
                            {explanation.context.time_of_day}
                        </span>
                        <span className="bg-netflix-red text-white px-2 py-1 rounded text-xs">
                            {explanation.context.device_type}
                        </span>
                    </div>
                </div>
            );
        }
        return null;
    };
    
    return (
        <div className="recommendation-explanation">
            <h3 className="text-lg font-bold mb-3">Why This Was Recommended</h3>
            
            {getCollaborativeExplanation()}
            {getContentBasedExplanation()}
            {getSequentialExplanation()}
            {getContextExplanation()}
        </div>
    );
};

// Export component to window object
window.Explanation = Explanation; 