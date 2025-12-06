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
const actualFilenames = new Set(actualFiles);

// Find files in download_info but not in downloads/
const missingFiles = downloadInfo.filter(entry => !actualFilenames.has(entry.filename));
console.log(`\n=== Files in download_information.json but NOT in downloads/ ===`);
console.log(`Total: ${missingFiles.length}`);
if (missingFiles.length > 0 && missingFiles.length <= 20) {
    missingFiles.forEach(entry => {
        console.log(`  - ${entry.filename} (${entry.genre}, page ${entry.page})`);
    });
} else if (missingFiles.length > 20) {
    console.log(`  (showing first 20)`);
    missingFiles.slice(0, 20).forEach(entry => {
        console.log(`  - ${entry.filename} (${entry.genre}, page ${entry.page})`);
    });
}

// Find files in downloads/ but not in download_info
const unrecordedFiles = actualFiles.filter(file => !recordedFilenames.has(file));
console.log(`\n=== Files in downloads/ but NOT in download_information.json ===`);
console.log(`Total: ${unrecordedFiles.length}`);
if (unrecordedFiles.length > 0 && unrecordedFiles.length <= 20) {
    unrecordedFiles.forEach(file => {
        console.log(`  - ${file}`);
    });
} else if (unrecordedFiles.length > 20) {
    console.log(`  (showing first 20)`);
    unrecordedFiles.slice(0, 20).forEach(file => {
        console.log(`  - ${file}`);
    });
}

// Summary
console.log(`\n=== Summary ===`);
console.log(`Download information entries: ${downloadInfo.length}`);
console.log(`Actual files in downloads/: ${actualFiles.length}`);
console.log(`Missing files: ${missingFiles.length}`);
console.log(`Unrecorded files: ${unrecordedFiles.length}`);
console.log(`Match: ${downloadInfo.length - missingFiles.length} files`);

// Check for duplicates in download_info
const filenameCounts = {};
downloadInfo.forEach(entry => {
    filenameCounts[entry.filename] = (filenameCounts[entry.filename] || 0) + 1;
});
const duplicates = Object.entries(filenameCounts).filter(([filename, count]) => count > 1);
if (duplicates.length > 0) {
    console.log(`\n=== Duplicate filenames in download_information.json ===`);
    console.log(`Total duplicates: ${duplicates.length}`);
    if (duplicates.length <= 10) {
        duplicates.forEach(([filename, count]) => {
            console.log(`  - ${filename}: appears ${count} times`);
        });
    } else {
        console.log(`  (showing first 10)`);
        duplicates.slice(0, 10).forEach(([filename, count]) => {
            console.log(`  - ${filename}: appears ${count} times`);
        });
    }
}

