import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const COLORS = [
  { name: "Ink", bgr: [112, 54, 10] },
  { name: "Pine", bgr: [28, 82, 24] },
  { name: "Wine", bgr: [18, 18, 132] },
  { name: "Ochre", bgr: [0, 92, 150] },
  { name: "Onyx", bgr: [24, 24, 24] },
];

const TOOLS = [
  { name: "Pen", thickness: 4, glow: false },
  { name: "Marker", thickness: 9, glow: false },
  { name: "Pencil", thickness: 2, glow: false },
  { name: "Neon", thickness: 5, glow: true },
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
let previousPoint = null;
let selectedColor = COLORS[0];
let selectedTool = TOOLS[0];
let sizeScale = 1;
let sizeAdjustActive = false;
let lastColorSelectAt = 0;
let lastToolSelectAt = 0;
let lastScreenshotAt = 0;
let activeModeLabel = "Waiting for left hand";
let colorPaletteBoxes = [];
let toolPaletteBoxes = [];

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
  ctx.strokeStyle = "rgba(225,225,225,0.96)";
  ctx.lineWidth = thickness + 2;
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
    drawingCtx.shadowColor = bgrToCss(color, 0.45);
    drawingCtx.shadowBlur = thickness * 3;
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

    stageCtx.fillStyle = bgrToCss(choice.bgr.map((channel) => Math.min(channel + 6, 255)));
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
      if (kind === "color" && now - lastColorSelectAt >= 350) {
        selectedColor = item.choice;
        lastColorSelectAt = now;
        setStatus(`Color changed to ${selectedColor.name}`);
        return;
      }
      if (kind === "tool" && now - lastToolSelectAt >= 350) {
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
  if (now - lastScreenshotAt < 1600) return;
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

function updateSizeFromRightHand(rightHand) {
  sizeAdjustActive = false;
  if (!rightHand) return;

  const states = getFingerStates(rightHand);
  const isSizeMode = states.index && !states.middle && !states.ring && !states.pinky;
  if (!isSizeMode) return;

  sizeAdjustActive = true;
  const thumbTip = landmarkToPoint(rightHand[4]);
  const indexTip = landmarkToPoint(rightHand[8]);
  const distance = pointDistance(thumbTip, indexTip);
  const mappedScale = 0.45 + ((Math.max(10, Math.min(180, distance)) - 10) / 170) * (3.2 - 0.45);
  sizeScale = sizeScale * 0.7 + mappedScale * 0.3;

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
  stageCtx.fillStyle = "rgba(20,20,20,0.42)";
  stageCtx.fillRect(18, 18, 540, 138);
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
    activeModeLabel = "Waiting for left hand";
    return;
  }

  const indexTip = landmarkToPoint(leftHand[8]);
  const drawPoint = smoothPoint(indexTip);
  const brushThickness = getToolThickness();
  const eraserThickness = getEraserThickness();

  if (gesture === "draw") {
    activeModeLabel = "Draw";
    drawPointer(drawPoint, selectedColor.bgr, getPointerRadius(brushThickness));
    if (previousPoint) {
      drawSegment(previousPoint, drawPoint, selectedTool, selectedColor.bgr, brushThickness);
    }
    previousPoint = drawPoint;
    return;
  }

  if (gesture === "erase") {
    activeModeLabel = "Erase";
    drawPointer(drawPoint, [255, 255, 255], getPointerRadius(eraserThickness));
    if (previousPoint) {
      eraseSegment(previousPoint, drawPoint, eraserThickness);
    }
    previousPoint = drawPoint;
    return;
  }

  previousPoint = null;

  if (gesture === "color") {
    activeModeLabel = "Color select";
    const points = [landmarkToPoint(leftHand[12])];
    applyPaletteSelection(points, colorPaletteBoxes, "color");
    points.forEach((point) => drawPointer(point, [255, 255, 255], 8));
    return;
  }

  if (gesture === "tool") {
    activeModeLabel = "Tool select";
    const points = [landmarkToPoint(leftHand[16])];
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

  let result = null;
  if (video.currentTime !== lastVideoTime) {
    try {
      result = handLandmarker.detectForVideo(video, performance.now());
    } catch (error) {
      result = handLandmarker.detectForVideo(video);
    }
    lastVideoTime = video.currentTime;
  }

  const { left, right } = getHandsByLabel(result);
  if (left) drawHandMesh(left, "Left");
  if (right) drawHandMesh(right, "Right");

  const leftStates = left ? getFingerStates(left) : null;
  const gesture = leftStates ? classifyGesture(leftStates) : "idle";

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
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
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
