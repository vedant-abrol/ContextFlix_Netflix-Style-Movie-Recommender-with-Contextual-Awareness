// SearchBox.jsx - Component for searching movies

const SearchBox = ({ onSearch }) => {
    const [searchTerm, setSearchTerm] = React.useState('');

    const handleInputChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        if (onSearch) {
            onSearch(searchTerm);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="w-full max-w-md mx-auto my-4">
            <div className="relative">
                <input 
                    type="text"
                    value={searchTerm}
                    onChange={handleInputChange}
                    placeholder="Search for movies, series, genres..."
                    className="w-full py-2 px-4 pr-10 bg-netflix-dark-gray text-white border border-gray-700 rounded-md focus:ring-netflix-red focus:border-netflix-red focus:outline-none"
                />
                <button 
                    type="submit"
                    className="absolute right-0 top-0 h-full px-3 text-gray-400 hover:text-netflix-red"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                </button>
            </div>
        </form>
    );
}; 