import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const COLORS = [
  { name: "Onyx", bgr: [18, 18, 18] },
  { name: "Pine", bgr: [20, 74, 28] },
  { name: "Merlot", bgr: [28, 28, 118] },
  { name: "Umber", bgr: [18, 64, 102] },
  { name: "Navy", bgr: [88, 48, 18] },
];

const TOOLS = [
  { name: "Pen", thickness: 4, glow: false },
  { name: "Marker", thickness: 9, glow: false },
  { name: "Pencil", thickness: 2, glow: false },
  { name: "Brush", thickness: 6, glow: false },
];

const FINGER_LANDMARKS = {
  index: [5, 6, 8],
  middle: [9, 10, 12],
  ring: [13, 14, 16],
  pinky: [17, 18, 20],
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];

const STABILITY = {
  handHoldMs: 180,
  gestureConfirmFrames: 4,
  idleConfirmFrames: 2,
  sizeModeConfirmFrames: 3,
  drawSmoothing: 0.36,
  paletteSmoothing: 0.28,
  sizeSmoothing: 0.2,
  sizeScaleSmoothing: 0.16,
  sizeScaleStep: 0.05,
  sizeScaleMin: 0.55,
  sizeScaleMax: 2.4,
  pinchMinDistance: 10,
  pinchMaxDistance: 170,
  paletteCooldownMs: 350,
  screenshotCooldownMs: 1600,
  drawMinDistance: 1.1,
};

const video = document.getElementById("camera");
const stage = document.getElementById("stage");
const stageCtx = stage.getContext("2d");
const drawingLayer = document.createElement("canvas");
const drawingCtx = drawingLayer.getContext("2d");

const startButton = document.getElementById("startButton");
const clearButton = document.getElementById("clearButton");
const shotButton = document.getElementById("shotButton");
const statusText = document.getElementById("statusText");
const toolMetric = document.getElementById("toolMetric");
const colorMetric = document.getElementById("colorMetric");
const sizeMetric = document.getElementById("sizeMetric");

let handLandmarker = null;
let stream = null;
let animationFrameId = 0;
let lastVideoTime = -1;
let latestDetection = null;
let previousPoint = null;
let selectedColor = COLORS[0];
let selectedTool = TOOLS[0];
let sizeScale = 1;
let sizeAdjustActive = false;
let sizeModeFrames = 0;
let lastColorSelectAt = 0;
let lastToolSelectAt = 0;
let lastScreenshotAt = 0;
let activeModeLabel = "Waiting for left hand";
let colorPaletteBoxes = [];
let toolPaletteBoxes = [];
const trackedPoints = new Map();
const handCache = {
  left: { landmarks: null, lastSeenAt: 0 },
  right: { landmarks: null, lastSeenAt: 0 },
};
const gestureTracker = {
  candidate: "idle",
  stable: "idle",
  frames: 0,
};

function setStatus(message) {
  statusText.textContent = message;
}

function resizeCanvases() {
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  if (stage.width === width && stage.height === height) {
    return;
  }

  const oldDrawing = document.createElement("canvas");
  oldDrawing.width = drawingLayer.width || width;
  oldDrawing.height = drawingLayer.height || height;
  const oldCtx = oldDrawing.getContext("2d");
  oldCtx.drawImage(drawingLayer, 0, 0);

  stage.width = width;
  stage.height = height;
  drawingLayer.width = width;
  drawingLayer.height = height;
  drawingCtx.clearRect(0, 0, width, height);
  drawingCtx.drawImage(oldDrawing, 0, 0, width, height);
}

function bgrToCss([b, g, r], alpha = 1) {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function pointDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleBetween(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const denominator = Math.hypot(ba.x, ba.y) * Math.hypot(bc.x, bc.y);
  if (!denominator) return 0;
  const cosine = Math.max(-1, Math.min(1, (ba.x * bc.x + ba.y * bc.y) / denominator));
  return Math.acos(cosine) * (180 / Math.PI);
}

function landmarkToPoint(landmark) {
  return {
    x: stage.width - landmark.x * stage.width,
    y: landmark.y * stage.height,
  };
}

function smoothTrackedPoint(key, point, smoothing) {
  if (!point) {
    trackedPoints.delete(key);
    return null;
  }

  const previous = trackedPoints.get(key);
  const next = !previous
    ? point
    : {
        x: previous.x * (1 - smoothing) + point.x * smoothing,
        y: previous.y * (1 - smoothing) + point.y * smoothing,
      };

  trackedPoints.set(key, next);
  return next;
}

function clearTrackedPoints(keys) {
  keys.forEach((key) => trackedPoints.delete(key));
}

function getFingerStates(landmarks) {
  const wrist = landmarkToPoint(landmarks[0]);
  const states = {};

  for (const [name, [mcpIdx, pipIdx, tipIdx]] of Object.entries(FINGER_LANDMARKS)) {
    const mcp = landmarkToPoint(landmarks[mcpIdx]);
    const pip = landmarkToPoint(landmarks[pipIdx]);
    const tip = landmarkToPoint(landmarks[tipIdx]);
    const isStraight = angleBetween(mcp, pip, tip) > 150;
    const reachesOut = pointDistance(wrist, tip) > pointDistance(wrist, pip) * 1.12;
    states[name] = isStraight && reachesOut;
  }

  return states;
}

function classifyGesture(fingerStates) {
  const { index, middle, ring, pinky } = fingerStates;
  if (index && !middle && !ring && !pinky) return "draw";
  if (middle && !index && !ring && !pinky) return "color";
  if (ring && !index && !middle && !pinky) return "tool";
  if (pinky && !index && !middle && !ring) return "screenshot";
  if (index && middle && ring && pinky) return "erase";
  return "idle";
}

function getToolThickness(tool = selectedTool) {
  return Math.max(1, Math.round(tool.thickness * sizeScale));
}

function getEraserThickness() {
  return Math.max(12, Math.round(24 * sizeScale));
}

function getPointerRadius(size) {
  return Math.max(4, Math.min(18, Math.floor(size / 2) + 2));
}

function drawVisibleLine(ctx, start, end, color, thickness) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(8, 8, 8, 0.96)";
  ctx.lineWidth = thickness + 3;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.strokeStyle = bgrToCss(color);
  ctx.lineWidth = thickness;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
}

function drawSegment(start, end, tool, color, thickness) {
  drawingCtx.save();
  if (tool.glow) {
    drawingCtx.shadowColor = "rgba(0, 0, 0, 0.28)";
    drawingCtx.shadowBlur = thickness * 1.75;
  }
  drawVisibleLine(drawingCtx, start, end, color, thickness);
  drawingCtx.restore();
}

function eraseSegment(start, end, thickness) {
  drawingCtx.save();
  drawingCtx.globalCompositeOperation = "destination-out";
  drawingCtx.lineCap = "round";
  drawingCtx.lineJoin = "round";
  drawingCtx.lineWidth = thickness;
  drawingCtx.beginPath();
  drawingCtx.moveTo(start.x, start.y);
  drawingCtx.lineTo(end.x, end.y);
  drawingCtx.stroke();
  drawingCtx.beginPath();
  drawingCtx.arc(end.x, end.y, Math.max(8, thickness / 2), 0, Math.PI * 2);
  drawingCtx.fill();
  drawingCtx.restore();
}

function drawPointer(point, color, radius = 6) {
  stageCtx.beginPath();
  stageCtx.strokeStyle = bgrToCss(color);
  stageCtx.lineWidth = 1.25;
  stageCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  stageCtx.stroke();
  stageCtx.beginPath();
  stageCtx.fillStyle = bgrToCss(color);
  stageCtx.arc(point.x, point.y, Math.max(2, Math.floor(radius / 3)), 0, Math.PI * 2);
  stageCtx.fill();
}

function drawHandMesh(landmarks, label) {
  const connectionColor = label === "Right" ? "rgba(120, 210, 255, 0.9)" : "rgba(178, 255, 196, 0.9)";
  for (const [start, end] of HAND_CONNECTIONS) {
    const a = landmarkToPoint(landmarks[start]);
    const b = landmarkToPoint(landmarks[end]);
    stageCtx.strokeStyle = connectionColor;
    stageCtx.lineWidth = 2;
    stageCtx.beginPath();
    stageCtx.moveTo(a.x, a.y);
    stageCtx.lineTo(b.x, b.y);
    stageCtx.stroke();
  }

  landmarks.forEach((landmark, index) => {
    const point = landmarkToPoint(landmark);
    const radius = [4, 8, 12, 16, 20].includes(index) ? 4 : 3;
    stageCtx.fillStyle = "rgba(255,255,255,0.95)";
    stageCtx.beginPath();
    stageCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    stageCtx.fill();
    stageCtx.strokeStyle = connectionColor;
    stageCtx.lineWidth = 1;
    stageCtx.beginPath();
    stageCtx.arc(point.x, point.y, radius + 2, 0, Math.PI * 2);
    stageCtx.stroke();
  });
}

function drawColorPalette(active) {
  colorPaletteBoxes = [];
  if (!active) return;

  const boxWidth = 108;
  const boxHeight = 52;
  const left = 28;
  const top = 98;
  const gap = 10;

  COLORS.forEach((choice, index) => {
    const x = left + index * (boxWidth + gap);
    colorPaletteBoxes.push({ rect: { x, y: top, width: boxWidth, height: boxHeight }, choice });

    stageCtx.fillStyle = bgrToCss(choice.bgr);
    stageCtx.fillRect(x, top, boxWidth, boxHeight);
    stageCtx.strokeStyle = choice.name === selectedColor.name ? "#ffffff" : "rgba(50,50,50,0.9)";
    stageCtx.lineWidth = choice.name === selectedColor.name ? 3 : 2;
    stageCtx.strokeRect(x, top, boxWidth, boxHeight);
    stageCtx.fillStyle = "#f5f5f5";
    stageCtx.font = '700 15px "Manrope", sans-serif';
    stageCtx.fillText(choice.name, x + 12, top + 31);
  });
}

function drawToolPalette(active) {
  toolPaletteBoxes = [];
  if (!active) return;

  const boxWidth = 126;
  const boxHeight = 50;
  const gap = 10;
  const totalWidth = TOOLS.length * boxWidth + (TOOLS.length - 1) * gap;
  const startX = Math.max((stage.width - totalWidth) / 2, 16);
  const top = stage.height - 86;

  TOOLS.forEach((tool, index) => {
    const x = startX + index * (boxWidth + gap);
    toolPaletteBoxes.push({ rect: { x, y: top, width: boxWidth, height: boxHeight }, tool });

    stageCtx.fillStyle = tool.name === selectedTool.name ? "rgba(70,88,112,0.96)" : "rgba(35,42,54,0.96)";
    stageCtx.fillRect(x, top, boxWidth, boxHeight);
    stageCtx.strokeStyle = "rgba(235,235,235,0.95)";
    stageCtx.lineWidth = 2;
    stageCtx.strokeRect(x, top, boxWidth, boxHeight);
    stageCtx.fillStyle = "#f5f5f5";
    stageCtx.font = '700 16px "Manrope", sans-serif';
    stageCtx.fillText(tool.name, x + 18, top + 31);
  });
}

function pointInsideRect(point, rect) {
  return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y && point.y <= rect.y + rect.height;
}

function applyPaletteSelection(points, items, kind) {
  const now = performance.now();
  for (const point of points) {
    for (const item of items) {
      if (!pointInsideRect(point, item.rect)) continue;
      if (kind === "color" && now - lastColorSelectAt >= STABILITY.paletteCooldownMs) {
        selectedColor = item.choice;
        lastColorSelectAt = now;
        setStatus(`Color changed to ${selectedColor.name}`);
        return;
      }
      if (kind === "tool" && now - lastToolSelectAt >= STABILITY.paletteCooldownMs) {
        selectedTool = item.tool;
        lastToolSelectAt = now;
        setStatus(`Tool changed to ${selectedTool.name}`);
        return;
      }
    }
  }
}

function updateMetrics() {
  toolMetric.textContent = selectedTool.name;
  colorMetric.textContent = selectedColor.name;
  sizeMetric.textContent = `${getToolThickness()} px`;
}

function downloadScreenshot() {
  const now = performance.now();
  if (now - lastScreenshotAt < STABILITY.screenshotCooldownMs) return;
  lastScreenshotAt = now;
  const link = document.createElement("a");
  link.href = stage.toDataURL("image/png");
  link.download = `airwriter-live-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
  link.click();
  setStatus("Screenshot saved.");
}

function clearCanvas() {
  drawingCtx.clearRect(0, 0, drawingLayer.width, drawingLayer.height);
  setStatus("Canvas cleared.");
}

function updateHandCache(result) {
  const now = performance.now();
  const detected = getHandsByLabel(result);

  for (const handName of ["left", "right"]) {
    const landmarks = detected[handName];
    if (landmarks) {
      handCache[handName].landmarks = landmarks;
      handCache[handName].lastSeenAt = now;
      continue;
    }

    if (now - handCache[handName].lastSeenAt > STABILITY.handHoldMs) {
      handCache[handName].landmarks = null;
    }
  }

  return {
    left: handCache.left.landmarks,
    right: handCache.right.landmarks,
  };
}

function updateStableGesture(nextGesture) {
  if (nextGesture === gestureTracker.candidate) {
    gestureTracker.frames += 1;
  } else {
    gestureTracker.candidate = nextGesture;
    gestureTracker.frames = 1;
  }

  const requiredFrames = nextGesture === "idle"
    ? STABILITY.idleConfirmFrames
    : STABILITY.gestureConfirmFrames;

  if (gestureTracker.frames >= requiredFrames) {
    gestureTracker.stable = nextGesture;
  }

  return gestureTracker.stable;
}

function updateSizeFromRightHand(rightHand) {
  sizeAdjustActive = false;
  if (!rightHand) {
    sizeModeFrames = 0;
    clearTrackedPoints(["right-thumb", "right-index"]);
    return;
  }

  const states = getFingerStates(rightHand);
  const wantsSizeMode = states.index && !states.middle && !states.ring && !states.pinky;
  sizeModeFrames = wantsSizeMode ? sizeModeFrames + 1 : 0;
  if (sizeModeFrames < STABILITY.sizeModeConfirmFrames) return;

  sizeAdjustActive = true;
  const thumbTip = smoothTrackedPoint("right-thumb", landmarkToPoint(rightHand[4]), STABILITY.sizeSmoothing);
  const indexTip = smoothTrackedPoint("right-index", landmarkToPoint(rightHand[8]), STABILITY.sizeSmoothing);
  const distance = pointDistance(thumbTip, indexTip);
  const clampedDistance = Math.max(STABILITY.pinchMinDistance, Math.min(STABILITY.pinchMaxDistance, distance));
  const mappedScale = STABILITY.sizeScaleMin
    + ((clampedDistance - STABILITY.pinchMinDistance) / (STABILITY.pinchMaxDistance - STABILITY.pinchMinDistance))
      * (STABILITY.sizeScaleMax - STABILITY.sizeScaleMin);
  const smoothedScale = sizeScale * (1 - STABILITY.sizeScaleSmoothing) + mappedScale * STABILITY.sizeScaleSmoothing;
  sizeScale = Math.round(smoothedScale / STABILITY.sizeScaleStep) * STABILITY.sizeScaleStep;

  stageCtx.strokeStyle = "rgba(255,215,140,0.95)";
  stageCtx.lineWidth = 2;
  stageCtx.beginPath();
  stageCtx.moveTo(thumbTip.x, thumbTip.y);
  stageCtx.lineTo(indexTip.x, indexTip.y);
  stageCtx.stroke();

  const previewRadius = getPointerRadius(getToolThickness());
  drawPointer(thumbTip, [245, 245, 245], previewRadius);
  drawPointer(indexTip, [245, 245, 245], previewRadius);
}

function renderHud() {
  stageCtx.fillStyle = "rgba(10, 10, 10, 0.58)";
  stageCtx.fillRect(18, 18, 560, 144);
  stageCtx.fillStyle = "#ffffff";
  stageCtx.font = '700 36px "Fraunces", serif';
  stageCtx.fillText("Air Writer Live", 34, 58);
  stageCtx.font = '600 20px "Manrope", sans-serif';
  stageCtx.fillText(`Mode: ${activeModeLabel}`, 34, 92);
  stageCtx.fillText(`Tool: ${selectedTool.name}`, 34, 122);

  stageCtx.font = '600 18px "Manrope", sans-serif';
  stageCtx.fillText("Left hand draws. Right thumb + index change size.", 590, 52);
  stageCtx.fillText(`Brush ${getToolThickness()} px`, 590, 82);
  stageCtx.fillText(`Eraser ${getEraserThickness()} px`, 590, 112);

  if (sizeAdjustActive) {
    stageCtx.fillStyle = "rgba(255,235,170,1)";
    stageCtx.fillText("Size adjust active", 820, 112);
    stageCtx.fillStyle = "#ffffff";
  }
}

function smoothPoint(point) {
  if (!previousPoint) return point;
  return {
    x: previousPoint.x * 0.5 + point.x * 0.5,
    y: previousPoint.y * 0.5 + point.y * 0.5,
  };
}

function getHandsByLabel(result) {
  const hands = { left: null, right: null };
  if (!result?.landmarks || !result?.handedness) {
    return hands;
  }

  result.landmarks.forEach((landmarks, index) => {
    const label = result.handedness[index]?.[0]?.categoryName;
    if (label === "Left") hands.left = landmarks;
    if (label === "Right") hands.right = landmarks;
  });

  return hands;
}

function handleLeftHand(leftHand, gesture) {
  if (!leftHand) {
    previousPoint = null;
    gestureTracker.candidate = "idle";
    gestureTracker.stable = "idle";
    gestureTracker.frames = 0;
    clearTrackedPoints(["left-index", "left-middle", "left-ring"]);
    activeModeLabel = "Waiting for left hand";
    return;
  }

  const indexTip = smoothTrackedPoint("left-index", landmarkToPoint(leftHand[8]), STABILITY.drawSmoothing);
  const middleTip = smoothTrackedPoint("left-middle", landmarkToPoint(leftHand[12]), STABILITY.paletteSmoothing);
  const ringTip = smoothTrackedPoint("left-ring", landmarkToPoint(leftHand[16]), STABILITY.paletteSmoothing);
  const drawPoint = smoothPoint(indexTip);
  const brushThickness = getToolThickness();
  const eraserThickness = getEraserThickness();

  if (gesture === "draw") {
    activeModeLabel = "Draw";
    drawPointer(drawPoint, selectedColor.bgr, getPointerRadius(brushThickness));
    if (previousPoint && pointDistance(previousPoint, drawPoint) >= STABILITY.drawMinDistance) {
      drawSegment(previousPoint, drawPoint, selectedTool, selectedColor.bgr, brushThickness);
    }
    previousPoint = drawPoint;
    return;
  }

  if (gesture === "erase") {
    activeModeLabel = "Erase";
    drawPointer(drawPoint, [255, 255, 255], getPointerRadius(eraserThickness));
    if (previousPoint && pointDistance(previousPoint, drawPoint) >= STABILITY.drawMinDistance) {
      eraseSegment(previousPoint, drawPoint, eraserThickness);
    }
    previousPoint = drawPoint;
    return;
  }

  previousPoint = null;

  if (gesture === "color") {
    activeModeLabel = "Color select";
    const points = [middleTip];
    applyPaletteSelection(points, colorPaletteBoxes, "color");
    points.forEach((point) => drawPointer(point, [255, 255, 255], 8));
    return;
  }

  if (gesture === "tool") {
    activeModeLabel = "Tool select";
    const points = [ringTip];
    applyPaletteSelection(points, toolPaletteBoxes, "tool");
    points.forEach((point) => drawPointer(point, [255, 255, 255], 8));
    return;
  }

  if (gesture === "screenshot") {
    activeModeLabel = "Screenshot";
    downloadScreenshot();
    return;
  }

  activeModeLabel = "Idle";
}

function drawSceneFrame() {
  stageCtx.save();
  stageCtx.translate(stage.width, 0);
  stageCtx.scale(-1, 1);
  stageCtx.drawImage(video, 0, 0, stage.width, stage.height);
  stageCtx.restore();

  stageCtx.drawImage(drawingLayer, 0, 0);
}

async function renderLoop() {
  if (!handLandmarker || video.readyState < 2) {
    animationFrameId = requestAnimationFrame(renderLoop);
    return;
  }

  resizeCanvases();
  drawSceneFrame();

  let result = latestDetection;
  if (video.currentTime !== lastVideoTime) {
    try {
      result = handLandmarker.detectForVideo(video, performance.now());
    } catch (error) {
      result = handLandmarker.detectForVideo(video);
    }
    latestDetection = result;
    lastVideoTime = video.currentTime;
  }

  const { left, right } = updateHandCache(result);
  if (left) drawHandMesh(left, "Left");
  if (right) drawHandMesh(right, "Right");

  const leftStates = left ? getFingerStates(left) : null;
  const gesture = leftStates ? updateStableGesture(classifyGesture(leftStates)) : "idle";

  updateSizeFromRightHand(right);
  drawColorPalette(gesture === "color");
  drawToolPalette(gesture === "tool");
  handleLeftHand(left, gesture);
  renderHud();
  updateMetrics();

  animationFrameId = requestAnimationFrame(renderLoop);
}

async function ensureLandmarker() {
  if (handLandmarker) return;
  setStatus("Loading hand tracking model...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./models/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.68,
    minTrackingConfidence: 0.7,
  });
}

async function startCamera() {
  startButton.disabled = true;
  try {
    await ensureLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvases();
    clearButton.disabled = false;
    shotButton.disabled = false;
    setStatus("Camera is live. Use your left hand to draw.");
    cancelAnimationFrame(animationFrameId);
    animationFrameId = requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error(error);
    setStatus("Could not start the camera. Please allow webcam access and reload.");
    startButton.disabled = false;
  }
}

startButton.addEventListener("click", startCamera);
clearButton.addEventListener("click", clearCanvas);
shotButton.addEventListener("click", downloadScreenshot);

window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(animationFrameId);
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
});
