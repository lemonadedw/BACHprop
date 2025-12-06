const fs = require('fs');
const path = require('path');

const downloadsDir = path.join(__dirname, '..', 'downloads');
const download_information_file = path.join(__dirname, '..', 'download_information.json');

// Read download information
let downloadInfo = [];
try {
    const data = fs.readFileSync(download_information_file, 'utf8');
    downloadInfo = JSON.parse(data);
    console.log(`Loaded ${downloadInfo.length} entries from download_information.json\n`);
} catch (error) {
    console.error(`Error reading download_information.json: ${error.message}`);
    process.exit(1);
}

// Get actual files in downloads directory
let actualFiles = [];
try {
    actualFiles = fs.readdirSync(downloadsDir)
        .filter(file => file.endsWith('.mid') || file.endsWith('.midi'))
        .sort();
    console.log(`Found ${actualFiles.length} files in downloads directory\n`);
} catch (error) {
    console.error(`Error reading downloads directory: ${error.message}`);
    process.exit(1);
}

// Create sets for comparison
const recordedFilenames = new Set(downloadInfo.map(entry => entry.filename));

// Find files in downloads/ but not in download_info
const unrecordedFiles = actualFiles.filter(file => !recordedFilenames.has(file));
console.log(`Found ${unrecordedFiles.length} unrecorded files\n`);

if (unrecordedFiles.length === 0) {
    console.log('No unrecorded files found. Nothing to repair!');
    process.exit(0);
}

// Add unrecorded files to download_info
console.log('Adding unrecorded files to download_information.json...\n');

let addedCount = 0;
unrecordedFiles.forEach((filename, index) => {
    // Try to extract info from filename or use defaults
    // Remove .mid extension for genre detection
    const nameWithoutExt = filename.replace(/\.mid$/i, '');
    
    // Try to guess genre from filename patterns (very basic)
    let guessedGenre = 'unknown';
    const genreKeywords = {
        'classic': ['bach', 'mozart', 'beethoven', 'chopin', 'classical'],
        'pop': ['pop', 'hit', 'song'],
        'rock': ['rock', 'guitar'],
        'jazz': ['jazz', 'swing'],
        'country': ['country'],
        'blues': ['blues'],
        'rap': ['rap', 'hip', 'hop'],
        'dance': ['dance', 'disco'],
        'punk': ['punk']
    };
    
    const lowerName = nameWithoutExt.toLowerCase();
    for (const [genre, keywords] of Object.entries(genreKeywords)) {
        if (keywords.some(keyword => lowerName.includes(keyword))) {
            guessedGenre = genre;
            break;
        }
    }
    
    // Add entry
    downloadInfo.push({
        url: `https://www.midiworld.com/download/${index + 10000}`, // Placeholder URL
        genre: guessedGenre,
        page: 0, // Unknown page
        id: downloadInfo.length + 1,
        filename: filename,
        repaired: true // Flag to indicate this was added by repair script
    });
    addedCount++;
    
    if ((index + 1) % 100 === 0) {
        console.log(`  Processed ${index + 1}/${unrecordedFiles.length} files...`);
    }
});

// Save updated download information
try {
    fs.writeFileSync(download_information_file, JSON.stringify(downloadInfo, null, 2));
    console.log(`\n✓ Successfully added ${addedCount} unrecorded files to download_information.json`);
    console.log(`Total entries: ${downloadInfo.length}`);
} catch (error) {
    console.error(`\n✗ Error saving download_information.json: ${error.message}`);
    process.exit(1);
}

// Verify
const recordedFilenamesAfter = new Set(downloadInfo.map(entry => entry.filename));
const stillUnrecorded = actualFiles.filter(file => !recordedFilenamesAfter.has(file));
console.log(`\nVerification: ${stillUnrecorded.length} files still unrecorded`);

if (stillUnrecorded.length > 0 && stillUnrecorded.length <= 10) {
    console.log('Remaining unrecorded files:');
    stillUnrecorded.forEach(file => console.log(`  - ${file}`));
}

