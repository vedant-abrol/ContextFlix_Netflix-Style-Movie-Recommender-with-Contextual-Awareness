// Explanation.jsx - Component to explain why a recommendation is made

const Explanation = ({ movie, recommendationType, context }) => {
    if (!movie || !recommendationType) {
        return null;
    }

    let explanationText = "We think you'll like this movie.";

    switch (recommendationType) {
        case 'user_preference':
            explanationText = `Because you like ${movie.genres ? movie.genres[0] : 'similar movies'}, we recommend "${movie.title}".`;
            break;
        case 'contextual':
            explanationText = `Based on your current context (watching in the ${context?.timeOfDay} on a ${context?.deviceType}), "${movie.title}" is a great choice.`;
            break;
        case 'popular':
            explanationText = `"${movie.title}" is popular right now. Check it out!`;
            break;
        case 'similar_to_watched':
            explanationText = `Since you watched a similar movie recently, you might also enjoy "${movie.title}".`;
            break;
        default:
            explanationText = `Enjoy "${movie.title}"!`;
    }

    return (
        <div className="bg-netflix-black bg-opacity-80 p-3 rounded-md my-2 text-sm text-gray-300">
            <p>
                <strong className="text-netflix-red">Why this recommendation?</strong> {explanationText}
            </p>
        </div>
    );
}; 