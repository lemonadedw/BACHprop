const { genre_urls} = require('./urls');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');
puppeteer.use(StealthPlugin());

// Create downloads directory if it doesn't exist
const downloadsDir = path.join(__dirname, '..', 'downloads');
if (!fs.existsSync(downloadsDir)) {
    fs.mkdirSync(downloadsDir, { recursive: true });
}
const download_information_file = path.join(__dirname, '..', 'download_information.json');
if (!fs.existsSync(download_information_file)) {
    fs.writeFileSync(download_information_file, JSON.stringify([], null, 2));
}

async function main() {
  const browser = await puppeteer.launch({
    headless: false, // Set to true to run in headless mode
    userDataDir: path.join(__dirname, '..', 'puppeteer-profile'),
  });

  let score_links = [];

  try {
    const page = await browser.newPage();
    
    // Configure downloads to go to ./downloads folder
    const client = await page.target().createCDPSession();
    await client.send('Page.setDownloadBehavior', {
        behavior: 'allow',
        downloadPath: downloadsDir
    });

    download_count = 0;
    download_information = [];
    let successfulDownloads = 0;

    for (const target of genre_urls) {
        genre = target.genre;
        url = target.url;
        max_pages = target.max_pages;

        console.log(`Scraping ${genre} | ${url}...`);

        for (let i = 1; i <= max_pages; i++) {
            // midiworld.com pagination format: /search/{page}/?q={genre}
            const pageUrl = i === 1 ? url : url.replace('/search/?q=', `/search/${i}/?q=`);
            await page.goto(pageUrl, { waitUntil: 'networkidle2' });
            await page.waitForSelector('ul li', { timeout: 10000 });
            // Extract links with filenames from <li> elements
            const linksWithNames = await page.$$eval('ul li', (listItems) => {
                return listItems.map(li => {
                    const link = li.querySelector('a[target="_blank"]');
                    if (!link || !link.href.includes('/download/')) {
                        return null;
                    }
                    // Extract filename from text before the download link
                    const text = li.textContent.trim();
                    const match = text.match(/^(.+?)\s*-\s*download/i);
                    const filename = match ? match[1].trim() : 'unknown';
                    return {
                        url: link.href,
                        filename: filename
                    };
                }).filter(item => item !== null);
            });
            info_embed_links = linksWithNames.map(({url, filename}) => ({
                url: url, 
                genre: genre, 
                page: i,
                filename: filename
                    .replace(/\s+/g, '_') // Replace spaces with underscores
                    .replace(/[<>:"/\\|?*]/g, '_') // Replace invalid filename characters
                    .replace(/_{2,}/g, '_') // Replace multiple underscores with single underscore
                    .replace(/^_+|_+$/g, '') // Remove leading/trailing underscores
            }));
            score_links.push(...info_embed_links);
        }
        console.log(`Current score links: ${score_links.length}`);
    }

    console.log(`Total score links: ${score_links.length}, beginning download...`);
    
    // Get list of files before downloads start
    const filesBeforeDownloads = new Set(fs.readdirSync(downloadsDir)
        .filter(file => file.endsWith('.mid') || file.endsWith('.midi')));
    
    total_download_count = score_links.length;
    const renameQueue = []; // Store {originalFilename, desiredFilename} pairs
    
    for (const {url, genre, page: pageNum, filename} of score_links) {
        try {
            // Visit the download link - midiworld.com automatically downloads
            // ERR_ABORTED is expected when a download starts (browser navigates to download URL)
            try {
                await page.goto(url, { waitUntil: 'networkidle2', timeout: 10000 });
            } catch (error) {
                // ERR_ABORTED means download started successfully
                if (error.message.includes('ERR_ABORTED') || error.message.includes('net::ERR_ABORTED')) {
                    // This is expected - download is happening, continue as success
                } else {
                    throw error; // Re-throw other errors
                }
            }
            
            // Wait a bit for download to start
            // await new Promise(resolve => setTimeout(resolve, 500));
            
            // Store the desired filename for later renaming
            const desiredFilename = filename + '.mid';
            renameQueue.push({
                url: url,
                genre: genre,
                page: pageNum,
                desiredFilename: desiredFilename
            });
            
            download_count++;
            successfulDownloads++;
            console.log(`[${download_count}/${total_download_count}] Downloaded from ${url}`);
        } catch (error) {
            console.log(`[${download_count + 1}/${total_download_count}] Skipping ${url} - ${error.message}`);
            continue;
        }
    }
    
    // Wait a bit more for all downloads to complete
    console.log('Waiting for downloads to complete...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Now rename all downloaded files
    try {
        const filesAfterDownloads = fs.readdirSync(downloadsDir)
            .filter(file => (file.endsWith('.mid') || file.endsWith('.midi')) && !filesBeforeDownloads.has(file))
            .map(file => {
                try {
                    return {
                        name: file,
                        time: fs.statSync(path.join(downloadsDir, file)).mtime.getTime()
                    };
                } catch (e) {
                    return null;
                }
            })
            .filter(file => file !== null)
            .sort((a, b) => a.time - b.time); // Oldest first (first downloaded = first in queue)
        
        console.log(`Renaming ${filesAfterDownloads.length} files...`);
        for (let i = 0; i < renameQueue.length && i < filesAfterDownloads.length; i++) {
            try {
                const {url, genre, page: pageNum, desiredFilename} = renameQueue[i];
                const originalFile = filesAfterDownloads[i].name;
                const oldPath = path.join(downloadsDir, originalFile);
                
                // Check if source file exists
                if (!fs.existsSync(oldPath)) {
                    console.log(`Skipping ${originalFile} - file not found`);
                    continue;
                }
                
                // Handle duplicate filenames
                let finalFilename = desiredFilename;
                let finalPath = path.join(downloadsDir, finalFilename);
                let counter = 1;
                while (fs.existsSync(finalPath)) {
                    const nameWithoutExt = desiredFilename.replace(/\.mid$/i, '');
                    finalFilename = `${nameWithoutExt}_${counter}.mid`;
                    finalPath = path.join(downloadsDir, finalFilename);
                    counter++;
                }
                
                fs.renameSync(oldPath, finalPath);
                download_information.push({
                    url: url,
                    genre: genre,
                    page: pageNum,
                    id: download_information.length + 1,
                    filename: finalFilename
                });
            } catch (error) {
                console.log(`Error renaming file ${i + 1}: ${error.message}`);
                continue;
            }
        }
        
        console.log(`Renamed ${download_information.length} files successfully`);
    } catch (error) {
        console.log(`Error during rename process: ${error.message}`);
    }

    // Save download information
    try {
        fs.writeFileSync(download_information_file, JSON.stringify(download_information, null, 2));
    } catch (error) {
        console.log(`Error saving download information: ${error.message}`);
    }

    console.log(`Downloaded ${successfulDownloads} scores, renamed ${download_information.length} files in '${downloadsDir}'`);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the main function
main();

