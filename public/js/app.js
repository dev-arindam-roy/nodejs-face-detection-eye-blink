// app.js
// Uses MediaPipe FaceMesh + CameraUtils
// Requirements: <video id="video">, <canvas id="overlay">, UI controls in index.html

const videoElement = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const statusEl = document.getElementById('status');
const timelineEl = document.getElementById('timeline');

const blinkCountEl = document.getElementById('blinkCount');
const mouthCountEl = document.getElementById('mouthCount');
const mouthNowEl = document.getElementById('mouthNow');
const yawDegEl = document.getElementById('yawDeg');
const turnDirEl = document.getElementById('turnDir');
const blinkRateEl = document.getElementById('blinkRate');

const earSlider = document.getElementById('ear');
const marSlider = document.getElementById('mar');
const smoothSlider = document.getElementById('smooth');
const debounceSlider = document.getElementById('debounce');

const earVal = document.getElementById('earVal');
const marVal = document.getElementById('marVal');
const smoothVal = document.getElementById('smoothVal');
const debounceVal = document.getElementById('debounceVal');

earVal.innerText = earSlider.value;
marVal.innerText = marSlider.value;
smoothVal.innerText = smoothSlider.value;
debounceVal.innerText = debounceSlider.value;

earSlider.oninput = () => earVal.innerText = earSlider.value;
marSlider.oninput = () => marVal.innerText = marSlider.value;
smoothSlider.oninput = () => smoothVal.innerText = smoothSlider.value;
debounceSlider.oninput = () => debounceVal.innerText = debounceSlider.value;

const socket = io(); // emits 'gesture' like earlier server

// --- Landmark index groups (MediaPipe FaceMesh) ---
// eye indices commonly used (from MediaPipe examples/tutorials).
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];   // right eye (medipipe indexing)
const LEFT_EYE  = [362, 385, 387, 263, 373, 380];  // left eye

// lips / mouth groups (upper and lower outer lip) — used to compute MAR
const UPPER_LIP = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78];
const LOWER_LIP = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61];

// nose key indices (we will average several nose landmarks to get stable nose point)
const NOSE_TIP_IDS = [1, 4, 5]; // commonly used nose-tip related indices

// util: get pixel from normalized landmark
function toPoint(lm, width, height){
  return { x: lm.x * width, y: lm.y * height, z: lm.z || 0 };
}
function avgPoints(points){
  const s = points.reduce((acc,p)=>({x:acc.x+p.x, y:acc.y+p.y}), {x:0,y:0});
  return { x: s.x / points.length, y: s.y / points.length };
}
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }

// EAR (eye aspect ratio) using 6 landmarks (same formula as classic EAR)
function eyeAspectRatio(eyePts){
  // expects array of 6 points: p0..p5
  const A = dist(eyePts[1], eyePts[5]);
  const B = dist(eyePts[2], eyePts[4]);
  const C = dist(eyePts[0], eyePts[3]);
  return (A + B) / (2.0 * C);
}

// MAR using upper/lower lip centroids and lip corners (approx)
function mouthAspectRatio(upperPts, lowerPts){
  const up = avgPoints(upperPts);
  const lo = avgPoints(lowerPts);
  // horizontal baseline: distance between left-most and right-most lip outer corners (approx)
  // find leftmost and rightmost x among combined lips
  const combined = upperPts.concat(lowerPts);
  let left = combined[0], right = combined[0];
  combined.forEach(p => { if (p.x < left.x) left = p; if (p.x > right.x) right = p; });
  const vert = dist(up, lo);
  const hor = dist(left, right) || 1;
  return vert / hor;
}

// Smoothing buffers (sliding windows)
let earHistory = [];
let marHistory = [];
function pushAndAvg(hist, value, maxLen){
  hist.push(value);
  while(hist.length > maxLen) hist.shift();
  return hist.reduce((s,v)=>s+v,0)/hist.length;
}

// event state
let blinkCount = 0;
let mouthCount = 0;
let mouthOpen = false;
let blinkLock = false;

let lastBlinksTimestamps = [];

// function to append timeline
function addTimeline(msg){
  const el = document.createElement('div');
  el.textContent = `${new Date().toLocaleTimeString()} — ${msg}`;
  timelineEl.prepend(el);
  // keep only latest 200 entries
  while(timelineEl.children.length > 200) timelineEl.removeChild(timelineEl.lastChild);
}

// head yaw estimation (2D yaw approximation):
// compute vector from eye-midpoint to nose; measure angle relative to camera x-axis.
// Positive yaw ~ turn to user's left; convert to degrees.
function computeYaw(leftEyeCenter, rightEyeCenter, nosePoint){
  const eyeMid = { x: (leftEyeCenter.x + rightEyeCenter.x)/2, y: (leftEyeCenter.y + rightEyeCenter.y)/2 };
  // vector from eyeMid to nose
  const vx = nosePoint.x - eyeMid.x;
  const vy = nosePoint.y - eyeMid.y;
  // yaw angle ~ arctangent of vx (x offset) normalized by eye distance
  const eyeDist = dist(leftEyeCenter, rightEyeCenter) || 1;
  const normalizedX = vx / eyeDist;
  const yawRad = Math.atan2(normalizedX, 1); // small-angle approx
  const yawDeg = yawRad * (180/Math.PI);
  return { yawDeg, normalizedX };
}

// MediaPipe setup
const faceMesh = new FaceMesh({ locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true, // includes iris landmarks; optional
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});

faceMesh.onResults(onResults);

// Camera utils
const camera = new Camera(videoElement, {
  onFrame: async () => { await faceMesh.send({image: videoElement}); },
  width: 640,
  height: 480
});

overlay.width = 640;
overlay.height = 480;
camera.start();
statusEl.innerText = 'camera started, loading model...';

function onResults(results) {
  ctx.clearRect(0,0,overlay.width, overlay.height);

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0){
    statusEl.innerText = 'no face';
    // small red overlay
    ctx.fillStyle = 'rgba(200,20,20,0.08)';
    ctx.fillRect(0,0,overlay.width, overlay.height);
    return;
  }

  statusEl.innerText = 'face detected';
  const landmarks = results.multiFaceLandmarks[0];

  const W = overlay.width, H = overlay.height;

  // gather required landmark groups in pixel coordinates
  const rightEyePts = RIGHT_EYE.map(i => toPoint(landmarks[i], W, H));
  const leftEyePts  = LEFT_EYE.map(i => toPoint(landmarks[i], W, H));

  const upperLipPts = UPPER_LIP.map(i => toPoint(landmarks[i], W, H));
  const lowerLipPts = LOWER_LIP.map(i => toPoint(landmarks[i], W, H));

  const nosePts = NOSE_TIP_IDS.map(i => toPoint(landmarks[i], W, H));
  const nosePt = avgPoints(nosePts);

  // compute per-frame raw EAR & MAR
  const leftEAR = eyeAspectRatio(leftEyePts);
  const rightEAR = eyeAspectRatio(rightEyePts);
  const avgEAR = (leftEAR + rightEAR) / 2.0;
  const marRaw = mouthAspectRatio(upperLipPts, lowerLipPts);

  // smoothing / sliding window
  const smoothWindow = parseInt(smoothSlider.value, 10);
  const earAvg = pushAndAvg(earHistory, avgEAR, smoothWindow);
  const marAvg = pushAndAvg(marHistory, marRaw, smoothWindow);

  // thresholds from UI
  const EAR_THRESHOLD = parseFloat(earSlider.value);
  const MAR_THRESHOLD = parseFloat(marSlider.value);
  const DEBOUNCE_FRAMES = parseInt(debounceSlider.value, 10);

  // Blink detection with debounce
  if (earAvg < EAR_THRESHOLD) {
    // eye closed frame
    if (!blinkLock) {
      // start lock counter
      blinkLock = { frames: 1 };
    } else {
      blinkLock.frames++;
    }

    if (blinkLock.frames === DEBOUNCE_FRAMES) {
      // a confirmed blink event
      blinkCount++;
      lastBlinksTimestamps.push(Date.now());
      addTimeline(`blink (EAR ${earAvg.toFixed(3)})`);
      socket.emit('gesture', { type: 'blink', ear: earAvg, timestamp: Date.now() });
    }
  } else {
    // reset blinkLock
    blinkLock = false;
  }

  // mouth open/close detection with hysteresis:
  if (marAvg > MAR_THRESHOLD) {
    if (!mouthOpen) {
      mouthOpen = true;
      mouthCount++;
      addTimeline(`mouth_open (MAR ${marAvg.toFixed(3)})`);
      socket.emit('gesture', { type: 'mouth_open', mar: marAvg, timestamp: Date.now() });
    }
  } else {
    if (mouthOpen) {
      mouthOpen = false;
      addTimeline(`mouth_close (MAR ${marAvg.toFixed(3)})`);
      socket.emit('gesture', { type: 'mouth_close', mar: marAvg, timestamp: Date.now() });
    }
  }

  // head yaw estimation
  // compute eye centers
  const leftEyeCenter = avgPoints(leftEyePts);
  const rightEyeCenter = avgPoints(rightEyePts);
  const yawRes = computeYaw(leftEyeCenter, rightEyeCenter, nosePt);
  const yawDeg = yawRes.yawDeg;
  const normalizedX = yawRes.normalizedX;

  // interpret turn
  const TURN_LEFT_THRESH = 0.12;
  const TURN_RIGHT_THRESH = -0.12;
  let turnDir = 'center';
  if (normalizedX > TURN_LEFT_THRESH) turnDir = 'left';
  else if (normalizedX < TURN_RIGHT_THRESH) turnDir = 'right';

  // emit head-turn events if large changes
  if (Math.abs(normalizedX) > 0.35) {
    socket.emit('gesture', { type: 'head_turn', dir: turnDir, yawDeg: yawDeg, timestamp: Date.now() });
  }

  // update UI stats
  blinkCountEl.innerText = blinkCount;
  mouthCountEl.innerText = mouthCount;
  mouthNowEl.innerText = mouthOpen ? 'yes' : 'no';
  yawDegEl.innerText = `${yawDeg.toFixed(1)}°`;
  turnDirEl.innerText = turnDir;

  // compute blink rate per minute (based on last 30s)
  const now = Date.now();
  lastBlinksTimestamps = lastBlinksTimestamps.filter(t => now - t < 60000); // last 60s
  const blinksPerMin = lastBlinksTimestamps.length;
  blinkRateEl.innerText = blinksPerMin;

  // draw overlay: draw small points for used landmarks and text
  ctx.lineWidth = 3;
  // draw eye contours
  ctx.strokeStyle = 'rgba(0,255,150,0.85)';
  drawPoly(ctx, leftEyePts);
  drawPoly(ctx, rightEyePts);

  // draw lips outline
  ctx.strokeStyle = 'rgba(255,150,0,0.85)';
  drawPoly(ctx, upperLipPts);
  drawPoly(ctx, lowerLipPts);

  // draw nose center
  ctx.fillStyle = 'rgba(120,180,255,0.95)';
  ctx.beginPath(); ctx.arc(nosePt.x, nosePt.y, 3.5, 0, Math.PI*2); ctx.fill();

  // draw readout box
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(8,8,240,84);
  ctx.fillStyle = '#9ff7d9';
  ctx.font = '12px monospace';
  ctx.fillText(`EAR(avg): ${earAvg.toFixed(3)}  (smoothed=${earAvg.toFixed(3)})`, 14, 26);
  ctx.fillText(`MAR: ${marAvg.toFixed(3)}`, 14, 46);
  ctx.fillText(`Blinks: ${blinkCount}  MouthCnt: ${mouthCount}`, 14, 66);
  ctx.fillText(`Yaw: ${yawDeg.toFixed(1)}°  (${turnDir})`, 14, 86);
}

function drawPoly(ctx, pts){
  if (!pts || pts.length===0) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  ctx.stroke();
}
