// Musescore URL configuration
// Musescore uses: https://musescore.com/sheetmusic/piano/{genre}
// Pagination: ?page={page_number}

module.exports = {
    genre_urls: [
        {
            genre: 'pop',
            url: 'https://musescore.com/sheetmusic/piano/pop',
            max_pages: 20
        },
        {
            genre: 'classical',
            url: 'https://musescore.com/sheetmusic/piano/classical',
            max_pages: 20
        },
        {
            genre: 'rock',
            url: 'https://musescore.com/sheetmusic/piano/rock',
            max_pages: 20
        },
        {
            genre: 'jazz',
            url: 'https://musescore.com/sheetmusic/piano/jazz',
            max_pages: 20
        },
        {
            genre: 'metal',
            url: 'https://musescore.com/sheetmusic/piano/metal',
            max_pages: 20
        },
        {
            genre: 'hip-hop',
            url: 'https://musescore.com/sheetmusic/piano/hip-hop',
            max_pages: 20
        },
        {
            genre: 'country',
            url: 'https://musescore.com/sheetmusic/piano/country',
            max_pages: 20
        },
    ]
}

