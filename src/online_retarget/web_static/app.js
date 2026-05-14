const form = document.getElementById("runForm");
const fileInput = document.getElementById("motionFile");
const runButton = document.getElementById("runButton");
const runMeta = document.getElementById("runMeta");
const stageGrid = document.getElementById("stageGrid");
const artifactList = document.getElementById("artifactList");
const detailText = document.getElementById("detailText");
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("frameSlider");
const frameMeta = document.getElementById("frameMeta");

let currentResult = null;
let currentView = "robot";

const sourceEdges = [
  ["Hips", "Spine1"],
  ["Spine1", "Spine2"],
  ["Spine2", "Chest"],
  ["Chest", "Neck1"],
  ["Neck1", "Head"],
  ["Chest", "LeftShoulder"],
  ["LeftShoulder", "LeftArm"],
  ["LeftArm", "LeftForeArm"],
  ["LeftForeArm", "LeftHand"],
  ["Chest", "RightShoulder"],
  ["RightShoulder", "RightArm"],
  ["RightArm", "RightForeArm"],
  ["RightForeArm", "RightHand"],
  ["Hips", "LeftLeg"],
  ["LeftLeg", "LeftShin"],
  ["LeftShin", "LeftFoot"],
  ["LeftFoot", "LeftToeBase"],
  ["Hips", "RightLeg"],
  ["RightLeg", "RightShin"],
  ["RightShin", "RightFoot"],
  ["RightFoot", "RightToeBase"],
];

const robotEdges = [
  ["pelvis", "torso_link"],
  ["torso_link", "head_link"],
  ["pelvis", "left_ankle_roll_link"],
  ["left_ankle_roll_link", "left_toe_link"],
  ["pelvis", "right_ankle_roll_link"],
  ["right_ankle_roll_link", "right_toe_link"],
  ["torso_link", "left_rubber_hand"],
  ["torso_link", "right_rubber_hand"],
];

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setDetail({ error: "Choose a BVH or SMPL-like file first." });
    return;
  }
  runButton.disabled = true;
  runMeta.textContent = "Running pipeline...";
  resetStages("Running");
  try {
    const data = new FormData();
    data.append("motion", file);
    const response = await fetch("/api/run", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Pipeline request failed");
    }
    applyResult(payload);
  } catch (error) {
    runMeta.textContent = "Run failed";
    setDetail({ error: String(error) });
  } finally {
    runButton.disabled = false;
  }
});

slider.addEventListener("input", () => drawFrame(Number(slider.value)));

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    currentView = button.dataset.view;
    document.querySelectorAll("[data-view]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    drawFrame(Number(slider.value));
  });
});

stageGrid.addEventListener("click", (event) => {
  const card = event.target.closest("[data-stage]");
  if (!card || !currentResult) {
    return;
  }
  const stage = currentResult.stages[card.dataset.stage];
  setDetail(stage || {});
});

function applyResult(result) {
  currentResult = result;
  runMeta.textContent = `${result.run_id} | ${result.input_format} | ${result.output_dir}`;
  updateStages(result.stages || {});
  updateArtifacts(result.artifacts || {});
  setDetail(result.stages || {});
  const frames = getFrames();
  slider.max = Math.max(0, frames.length - 1);
  slider.value = "0";
  drawFrame(0);
}

function resetStages(message) {
  document.querySelectorAll(".stage").forEach((card) => {
    card.className = "stage";
    card.querySelector("span").textContent = message;
  });
}

function updateStages(stages) {
  document.querySelectorAll(".stage").forEach((card) => {
    const stage = stages[card.dataset.stage] || {};
    card.className = `stage ${stage.status || ""}`;
    card.querySelector("span").textContent = stage.message || "No status";
  });
}

function updateArtifacts(artifacts) {
  artifactList.replaceChildren();
  Object.entries(artifacts).forEach(([key, value]) => {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = key;
    detail.textContent = value;
    artifactList.append(term, detail);
  });
}

function setDetail(value) {
  detailText.textContent = JSON.stringify(value, null, 2);
}

function getFrames() {
  if (!currentResult || !currentResult.preview) {
    return [];
  }
  const preview = currentResult.preview;
  if (currentView === "source") {
    return (preview.source && preview.source.frames) || [];
  }
  return (preview.robot && preview.robot.frames) || [];
}

function drawFrame(index) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  drawGround(width, height);

  const frames = getFrames();
  frameMeta.textContent = `Frame ${index}`;
  if (!frames.length) {
    drawEmpty("No preview frames available");
    return;
  }
  const frame = frames[Math.min(index, frames.length - 1)] || {};
  const edges = currentView === "source" ? sourceEdges : robotEdges;
  const points = Object.entries(frame)
    .filter(([, value]) => Array.isArray(value) && value.length >= 3)
    .map(([name, value]) => ({ name, xyz: value }));

  if (!points.length) {
    drawEmpty("No visible body points");
    return;
  }

  const projected = new Map();
  const bounds = projectBounds(points.map((point) => point.xyz));
  points.forEach((point) => {
    projected.set(point.name, project(point.xyz, bounds, width, height));
  });

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#245f73";
  edges.forEach(([from, to]) => {
    if (!projected.has(from) || !projected.has(to)) {
      return;
    }
    const a = projected.get(from);
    const b = projected.get(to);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  points.forEach((point) => {
    const p = projected.get(point.name);
    ctx.beginPath();
    ctx.fillStyle = keyPoint(point.name) ? "#7d6b2b" : "#1f7a4d";
    ctx.arc(p.x, p.y, keyPoint(point.name) ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawGround(width, height) {
  ctx.fillStyle = "#eef2ed";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8ded5";
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 46) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 46) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawEmpty(message) {
  ctx.fillStyle = "#68736b";
  ctx.font = "16px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function projectBounds(points) {
  const xs = points.map((point) => point[0] - point[1] * 0.35);
  const ys = points.map((point) => point[2] + point[1] * 0.18);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function project(point, bounds, width, height) {
  const isoX = point[0] - point[1] * 0.35;
  const isoY = point[2] + point[1] * 0.18;
  const spanX = Math.max(0.1, bounds.maxX - bounds.minX);
  const spanY = Math.max(0.1, bounds.maxY - bounds.minY);
  const scale = Math.min((width * 0.78) / spanX, (height * 0.72) / spanY);
  return {
    x: width / 2 + (isoX - (bounds.minX + bounds.maxX) / 2) * scale,
    y: height * 0.72 - (isoY - bounds.minY) * scale,
  };
}

function keyPoint(name) {
  return (
    name.includes("Head") ||
    name.includes("Hand") ||
    name.includes("Foot") ||
    name.includes("toe") ||
    name.includes("hand")
  );
}

drawFrame(0);
