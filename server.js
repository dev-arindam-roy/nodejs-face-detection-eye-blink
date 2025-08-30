const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const fs = require('fs');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = process.env.PORT || 3000;
const LOG_FILE = path.join(__dirname, 'events.log');

// serve static frontend
app.use(express.static(path.join(__dirname, 'public')));

// simple route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// socket.io connection
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('gesture', (payload) => {
    // payload example: { type: 'blink', value: 'left', timestamp: 169... }
    const logLine = `${new Date().toISOString()} | ${socket.id} | ${JSON.stringify(payload)}\n`;
    fs.appendFile(LOG_FILE, logLine, (err) => {
      if (err) console.error('Failed to write log', err);
    });
    // echo back to client if needed
    io.emit('gesture-broadcast', payload);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
