const fs = require('fs');
const path = require('path');
const { Midi } = require('@tonejs/midi');

const downloadsDir = path.join(__dirname, '..', 'downloads');
const download_information_file = path.join(__dirname, '..', 'download_information.json');

// Parse MIDI file and check if it has left/right hand indicators
function hasLeftRightHandTracks(filePath) {
    try {
        const buffer = fs.readFileSync(filePath);
        const midi = new Midi(buffer);
        
        // Must have exactly 2 tracks
        if (midi.tracks.length !== 2) {
            return false;
        }
        
        // Check if tracks have left/right hand indicators
        let leftHandFound = false;
        let rightHandFound = false;
        
        for (const track of midi.tracks) {
            const trackName = (track.name || '').toLowerCase();
            
            // Check for left hand indicators
            if (trackName.includes('left') || 
                trackName.includes('lh') ||
                trackName === 'l' ||
                trackName.includes('left hand')) {
                leftHandFound = true;
            }
            
            // Check for right hand indicators
            if (trackName.includes('right') || 
                trackName.includes('rh') ||
                trackName === 'r' ||
                trackName.includes('right hand')) {
                rightHandFound = true;
            }
        }
        
        // Both left and right hand must be found
        return leftHandFound && rightHandFound;
    } catch (error) {
        return false;
    }
}

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

// Count files with left/right hand tracks
let count = 0;

downloadInfo.forEach((entry, index) => {
    const filePath = path.join(downloadsDir, entry.filename);
    
    if (!fs.existsSync(filePath)) {
        return;
    }
    
    if (hasLeftRightHandTracks(filePath)) {
        count++;
    }
    
    if ((index + 1) % 100 === 0) {
        // Silent progress - no output
    }
});

// Output only the count
console.log(count);

