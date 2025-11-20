import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the public directory (videos, etc.)
app.use(express.static(path.join(__dirname, 'public')));

// Serve the React app build files
app.use(express.static(path.join(__dirname, 'dist')));

// API routes can be added here if needed
// app.use('/api', apiRoutes);

// For all other routes, serve the React app (SPA routing)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Serving React app and static assets`);
});

