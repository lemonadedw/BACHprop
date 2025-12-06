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
  
  // login first
  const page = await browser.newPage();
  await page.goto('https://musescore.com/login', { waitUntil: 'networkidle2' });
  console.log('Press Enter once you have logged in');
  await new Promise(resolve => process.stdin.once('data', () => resolve()));

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

    for (const target of genre_urls) {
        genre = target.genre;
        url = target.url;
        max_pages = target.max_pages;

        console.log(`Scraping ${genre} | ${url}...`);

        for (let i = 1; i <= max_pages; i++) {
            // Musescore pagination format: ?page={page_number}
            await page.goto(`${url}?page=${i}`, { waitUntil: 'networkidle2' });
            await page.waitForSelector('.VMT_w a', { timeout: 10000 });
            const links = (await page.$$eval('.VMT_w a', elements => 
                elements.map(element => element.href)
            )).filter(link => link.includes('/user/') && link.includes('/scores/')); // must be a score uploaded by a user
            info_embed_links = links.map(link => ({url: link, genre: genre, page: i}));
            score_links.push(...info_embed_links);
        }
        console.log(`Current score links: ${score_links.length}`);
    }

    console.log(`Total score links: ${score_links.length}, beginning download...`);
    total_download_count = score_links.length;
    for (const {url, genre, page: pageNum} of score_links) {
        try {
            await page.goto(url);
            await page.waitForNetworkIdle();
            await page.click('[name="download"]') // Download button to open midi option
            await page.waitForSelector('.diP_e.iCnHA', { timeout: 10000 });
            // Get all buttons within .KZ3mt
            const buttons = await page.$$('.KZ3mt button');
            
            // Find the button that contains "MIDI" text
            let midiButtonFound = false;
            for (const button of buttons) {
                const buttonText = await page.evaluate(el => el.textContent, button);
                if (buttonText.includes('MIDI')) {
                    await button.click();
                    midiButtonFound = true;
                    break;
                }
            }
            
            if (!midiButtonFound) {
                console.log(`[${download_count + 1}/${total_download_count}] MIDI Button for ${url} not found`);
            } else{
                console.log(`[${download_count}/${total_download_count}] Downloaded ${url}`);
                download_count++;
                download_information.push({
                    url: url,
                    genre: genre,
                    page: pageNum,
                    id: download_count
                });
            }
        } catch (error) {
            console.log(`[${download_count + 1}/${total_download_count}] Skipping ${url} - ${error.message}`);
            continue;
        }
    }

    fs.writeFileSync(download_information_file, JSON.stringify(download_information, null, 2));

    console.log(`Downloaded ${download_count} scores in '${downloadsDir}'`);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the main function
main();
