const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 4000;

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve exercise videos from public/videos
app.use('/videos', express.static(path.join(__dirname, 'public', 'videos')));

// Route for homepage
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Rectify frontend running at http://localhost:${PORT}`);
});

