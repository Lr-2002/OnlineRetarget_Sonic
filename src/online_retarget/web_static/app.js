const form = document.getElementById("runForm");
const fileInput = document.getElementById("motionFile");
const renderFramesInput = document.getElementById("renderFrames");
const compareRetargetersInput = document.getElementById("compareRetargeters");
const runButton = document.getElementById("runButton");
const runMeta = document.getElementById("runMeta");
const stageGrid = document.getElementById("stageGrid");
const artifactList = document.getElementById("artifactList");
const detailText = document.getElementById("detailText");
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");
const panelGrid = document.getElementById("panelGrid");
const video = document.getElementById("mujocoVideo");
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
    data.append("render_frames", renderFramesInput.checked ? "true" : "false");
    data.append("compare_retargeters", compareRetargetersInput.checked ? "true" : "false");
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
    updateVideo();
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
  updateVideo();
  drawFrame(0);
}

function resetStages(message) {
  video.pause();
  video.removeAttribute("src");
  video.style.display = "none";
  panelGrid.replaceChildren();
  panelGrid.style.display = "none";
  canvas.style.display = "block";
  slider.style.display = "block";
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

function updateVideo() {
  const panels =
    currentResult &&
    currentResult.preview &&
    Array.isArray(currentResult.preview.panels)
      ? currentResult.preview.panels
      : [];
  const videoUrl =
    currentResult &&
    currentResult.preview &&
    currentResult.preview.mujoco &&
    currentResult.preview.mujoco.video_url;
  const showPanels = currentView === "robot" && panels.length > 0;
  if (showPanels) {
    renderPanels(panels);
    panelGrid.style.display = "grid";
    video.style.display = "none";
    canvas.style.display = "none";
    slider.style.display = "none";
    frameMeta.textContent = panels.length > 1 ? "MuJoCo comparison" : "MuJoCo G1 render";
    return;
  }
  panelGrid.replaceChildren();
  panelGrid.style.display = "none";
  const showVideo = currentView === "robot" && Boolean(videoUrl);
  if (showVideo) {
    if (video.getAttribute("src") !== videoUrl) {
      video.src = videoUrl;
      video.load();
    }
    video.style.display = "block";
    canvas.style.display = "none";
    slider.style.display = "none";
    frameMeta.textContent = "MuJoCo G1 render";
    return;
  }
  video.pause();
  video.style.display = "none";
  canvas.style.display = "block";
  slider.style.display = "block";
}

function renderPanels(panels) {
  panelGrid.replaceChildren();
  panels.forEach((panel) => {
    const article = document.createElement("article");
    article.className = "render-panel";
    const title = document.createElement("h3");
    title.textContent = panel.title || panel.method || "Retargeter";
    const itemVideo = document.createElement("video");
    itemVideo.controls = true;
    itemVideo.playsInline = true;
    itemVideo.muted = true;
    itemVideo.src = panel.video_url || "";
    const status = document.createElement("p");
    status.textContent = `${panel.status || "unknown"} | ${panel.root_xy_locked ? "root XY locked" : "free root"}`;
    article.append(title, itemVideo, status);
    panelGrid.append(article);
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
  if (currentView === "robot" && video.style.display === "block") {
    return;
  }
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
  const edges = getEdges();
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

  drawEdges(projected, edges);

  points.forEach((point) => {
    const p = projected.get(point.name);
    ctx.beginPath();
    ctx.fillStyle = keyPoint(point.name) ? "#7d6b2b" : "#1f7a4d";
    ctx.arc(p.x, p.y, keyPoint(point.name) ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function getEdges() {
  if (
    currentView === "source" &&
    currentResult &&
    currentResult.preview &&
    currentResult.preview.source &&
    Array.isArray(currentResult.preview.source.capsule_edges)
  ) {
    return currentResult.preview.source.capsule_edges;
  }
  return currentView === "source" ? sourceEdges : robotEdges;
}

function drawEdges(projected, edges) {
  ctx.lineCap = currentView === "source" ? "round" : "butt";
  ctx.lineJoin = "round";
  ctx.lineWidth = currentView === "source" ? 18 : 3;
  ctx.strokeStyle = currentView === "source" ? "rgba(36, 95, 115, 0.34)" : "#245f73";
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
    if (currentView === "source") {
      ctx.lineWidth = 10;
      ctx.strokeStyle = "rgba(31, 122, 77, 0.58)";
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.lineWidth = 18;
      ctx.strokeStyle = "rgba(36, 95, 115, 0.34)";
    }
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
