// SearchBox.jsx - Search component for finding movies

const SearchBox = ({ onSearch, placeholder = "Search for movies..." }) => {
    const [searchTerm, setSearchTerm] = React.useState('');
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if (searchTerm.trim()) {
            onSearch(searchTerm);
        }
    };
    
    return (
        <form onSubmit={handleSubmit} className="search-box">
            <div className="relative">
                <input
                    type="text"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded-md pl-10 focus:outline-none focus:ring-1 focus:ring-netflix-red"
                    placeholder={placeholder}
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
                
                <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                </div>
                
                <button 
                    type="submit"
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-netflix-red text-white px-3 py-1 rounded-md text-sm"
                >
                    Search
                </button>
            </div>
        </form>
    );
};

// Export component to window object
window.SearchBox = SearchBox; 