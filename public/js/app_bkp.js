// app.js
const video = document.getElementById('inputVideo');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');

const socket = io();

// Detection parameters (tune these)
const EAR_THRESHOLD = 0.22;       // Eye aspect ratio below = closed (tweak)
const EAR_CONSEC_FRAMES = 2;     // number of consecutive frames to count a blink
const MAR_THRESHOLD = 0.55;      // Mouth aspect ratio above = open (tweak)
const MAR_CONSEC_FRAMES = 2;

let blinkCount = 0;
let mouthOpen = false;

let leftEyeClosedFrames = 0;
let rightEyeClosedFrames = 0;
let mouthOpenFrames = 0;

async function start() {
  // load models from CDN (face-api will auto fetch from CDN)
  const MODEL_URL = '/models';
  status('Loading models...');
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);

  status('Models loaded. Starting camera...');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 }, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      runDetectionLoop();
    };
  } catch (err) {
    status('Camera error: ' + err.message);
    console.error(err);
  }
}

function status(s) {
  statusEl.innerText = s;
}

function logEvent(obj) {
  const el = document.createElement('div');
  el.className = 'evt';
  el.textContent = `${new Date().toLocaleTimeString()} — ${obj.type} ${obj.detail || ''}`;
  logEl.prepend(el);
  // emit to server
  socket.emit('gesture', { ...obj, timestamp: Date.now() });
}

// utility functions: EAR and MAR
function euclidean(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// Eye Aspect Ratio: compares vertical eye landmarks to horizontal
function eyeAspectRatio(eye) {
  // eye: array of 6 points
  const A = euclidean(eye[1], eye[5]);
  const B = euclidean(eye[2], eye[4]);
  const C = euclidean(eye[0], eye[3]);
  return (A + B) / (2.0 * C);
}

// Mouth Aspect Ratio: distance of vertical vs horizontal
function mouthAspectRatio(mouth) {
  // mouth: 8 outer mouth points usually 48..59 in 68-landmarks
  const A = euclidean(mouth[2], mouth[10]); // 51-59
  const B = euclidean(mouth[4], mouth[8]);  // 53-57
  const C = euclidean(mouth[0], mouth[6]);  // 49-55
  return (A + B) / (2.0 * C);
}

async function runDetectionLoop() {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
  status('Running detection');
  // loop
  setInterval(async () => {
    if (video.paused || video.ended) return;
    const result = await faceapi.detectSingleFace(video, options).withFaceLandmarks(true);
    ctx.clearRect(0,0,overlay.width, overlay.height);

    if (result && result.landmarks) {
      const landmarks = result.landmarks;
      const leftEye = landmarks.getLeftEye();   // array of 6 pts
      const rightEye = landmarks.getRightEye();
      const mouth = landmarks.getMouth();       // array of pts, use outer

      // draw landmarks
      faceapi.draw.drawFaceLandmarks(overlay, result);

      const leftEAR = eyeAspectRatio(leftEye);
      const rightEAR = eyeAspectRatio(rightEye);
      const avgEAR = (leftEAR + rightEAR) / 2.0;
      const mar = mouthAspectRatio(mouth);

      // Eye detection (blink/close)
      if (leftEAR < EAR_THRESHOLD) leftEyeClosedFrames++; else leftEyeClosedFrames = 0;
      if (rightEAR < EAR_THRESHOLD) rightEyeClosedFrames++; else rightEyeClosedFrames = 0;

      // If both eyes closed for several consecutive frames -> blink/closed
      if (leftEyeClosedFrames >= EAR_CONSEC_FRAMES && rightEyeClosedFrames >= EAR_CONSEC_FRAMES) {
        // count blink once when threshold crossed
        if (!blinkLock) {
          blinkLock = true;
          blinkCount++;
          logEvent({ type: 'blink', detail: `count=${blinkCount}`, ears: { left: leftEAR.toFixed(3), right: rightEAR.toFixed(3) }});
        }
      } else {
        blinkLock = false;
      }

      // Mouth open detection
      if (mar > MAR_THRESHOLD) {
        mouthOpenFrames++;
      } else {
        mouthOpenFrames = 0;
      }

      if (mouthOpenFrames >= MAR_CONSEC_FRAMES) {
        if (!mouthOpen) {
          mouthOpen = true;
          logEvent({ type: 'mouth_open', detail: `mar=${mar.toFixed(3)}` });
        }
      } else {
        if (mouthOpen) {
          mouthOpen = false;
          logEvent({ type: 'mouth_close', detail: `mar=${mar.toFixed(3)}` });
        }
      }

      // overlay text
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(8,8,220,72);
      ctx.fillStyle = '#0f0';
      ctx.font = '14px monospace';
      ctx.fillText(`EAR L:${leftEAR.toFixed(3)} R:${rightEAR.toFixed(3)} AVG:${avgEAR.toFixed(3)}`, 12, 28);
      ctx.fillText(`MAR:${mar.toFixed(3)}`, 12, 50);
      ctx.fillText(`Blinks:${blinkCount}`, 12, 72);
    } else {
      ctx.fillStyle = 'rgba(255,0,0,0.2)';
      ctx.fillRect(0,0,overlay.width, overlay.height);
    }
  }, 100); // 10 FPS approx
}

// small locks / state
let blinkLock = false;

start();
