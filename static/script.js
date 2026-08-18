/* ============================================================
   Flappy AI — frontend game engine
   ------------------------------------------------------------
   This file renders and runs the LIVE, visible game in the
   browser. Its physics constants intentionally match
   game_logic.py exactly, so a bird trained by the Python
   Genetic Algorithm behaves the same way here.

   Two ways to control the bird:
     1. MANUAL  -> the player presses Space / taps the screen
     2. AI      -> weights fetched from the Flask backend
                   (/api/train_generation) decide every frame
   ============================================================ */

// ---------------- Constants (mirrors game_logic.py) ----------------
const GAME_WIDTH = 480;
const GAME_HEIGHT = 640;
const BIRD_X = 80;
const BIRD_RADIUS = 14;
const GRAVITY = 0.5;
const FLAP_STRENGTH = -8.0;
const PIPE_WIDTH = 60;
const PIPE_GAP = 160;
const PIPE_SPEED = 3.5;
const PIPE_SPACING = 220;

// ---------------- DOM references ----------------
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const startOverlay = document.getElementById("startOverlay");
const gameOverOverlay = document.getElementById("gameOverOverlay");
const finalScoreText = document.getElementById("finalScoreText");

const startBtn = document.getElementById("startBtn");
const restartBtn = document.getElementById("restartBtn");
const restartBtnOverlay = document.getElementById("restartBtnOverlay");
const aiModeBtn = document.getElementById("aiModeBtn");
const resetAiBtn = document.getElementById("resetAiBtn");
const modePill = document.getElementById("modePill");

const scoreValue = document.getElementById("scoreValue");
const bestScoreValue = document.getElementById("bestScoreValue");
const generationValue = document.getElementById("generationValue");
const aiBestValue = document.getElementById("aiBestValue");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// ---------------- Game state ----------------
let bird, pipes, score, alive, started;
let mode = "manual";          // "manual" | "ai"
let aiWeights = null;         // [w1, w2, w3, w4, bias] from the backend
let aiTrainingBusy = false;   // true while waiting for the server
let bestScore = parseInt(localStorage.getItem("flappyBestScore") || "0", 10);
bestScoreValue.textContent = bestScore;

let animationId = null;

// ---------------- Entities ----------------
function newBird() {
  return { y: GAME_HEIGHT / 2, velocity: 0 };
}

function newPipe(x) {
  const margin = 80;
  const gapY = margin + Math.random() * (GAME_HEIGHT - margin * 2);
  return { x, gapY, passed: false };
}

function resetGame() {
  bird = newBird();
  pipes = [newPipe(GAME_WIDTH + 100)];
  score = 0;
  alive = true;
  scoreValue.textContent = score;
}

// ---------------- Physics / update ----------------
function flap() {
  bird.velocity = FLAP_STRENGTH;
}

function nextPipe() {
  for (const p of pipes) {
    if (p.x + PIPE_WIDTH > BIRD_X) return p;
  }
  return pipes[pipes.length - 1];
}

function getState() {
  const np = nextPipe();
  const birdYNorm = bird.y / GAME_HEIGHT;
  const velNorm = Math.max(-1, Math.min(1, bird.velocity / 10));
  const distNorm = Math.max(0, (np.x - BIRD_X) / GAME_WIDTH);
  const gapYNorm = np.gapY / GAME_HEIGHT;
  return [birdYNorm, velNorm, distNorm, gapYNorm];
}

function aiDecide(state) {
  if (!aiWeights) return 0;
  let total = aiWeights[4]; // bias
  for (let i = 0; i < 4; i++) total += aiWeights[i] * state[i];
  return total > 0 ? 1 : 0;
}

function update() {
  if (!alive) return;

  // Decide whether to flap (AI mode only — manual mode uses key/click events)
  if (mode === "ai") {
    const action = aiDecide(getState());
    if (action === 1) flap();
  }

  bird.velocity += GRAVITY;
  bird.y += bird.velocity;

  for (const p of pipes) p.x -= PIPE_SPEED;

  if (pipes[pipes.length - 1].x < GAME_WIDTH - PIPE_SPACING) {
    pipes.push(newPipe(GAME_WIDTH + 20));
  }
  pipes = pipes.filter((p) => p.x > -PIPE_WIDTH);

  for (const p of pipes) {
    if (!p.passed && p.x + PIPE_WIDTH < BIRD_X) {
      p.passed = true;
      score += 1;
      scoreValue.textContent = score;
    }
  }

  // Collisions: floor / ceiling
  if (bird.y > GAME_HEIGHT - BIRD_RADIUS || bird.y < BIRD_RADIUS) {
    alive = false;
  }

  // Collisions: pipes
  const np = nextPipe();
  const top = np.gapY - PIPE_GAP / 2;
  const bottom = np.gapY + PIPE_GAP / 2;
  if (np.x < BIRD_X + BIRD_RADIUS && np.x + PIPE_WIDTH > BIRD_X - BIRD_RADIUS) {
    if (bird.y - BIRD_RADIUS < top || bird.y + BIRD_RADIUS > bottom) {
      alive = false;
    }
  }

  if (!alive) onGameOver();
}

// ---------------- Drawing ----------------
function draw() {
  ctx.clearRect(0, 0, GAME_WIDTH, GAME_HEIGHT);

  // Pipes
  for (const p of pipes) {
    const top = p.gapY - PIPE_GAP / 2;
    const bottom = p.gapY + PIPE_GAP / 2;

    const grad = ctx.createLinearGradient(p.x, 0, p.x + PIPE_WIDTH, 0);
    grad.addColorStop(0, "#2fae86");
    grad.addColorStop(1, "#4ce0d2");
    ctx.fillStyle = grad;

    ctx.fillRect(p.x, 0, PIPE_WIDTH, top);
    ctx.fillRect(p.x, bottom, PIPE_WIDTH, GAME_HEIGHT - bottom);

    ctx.fillStyle = "#1c8f6d";
    ctx.fillRect(p.x - 3, top - 16, PIPE_WIDTH + 6, 16);
    ctx.fillRect(p.x - 3, bottom, PIPE_WIDTH + 6, 16);
  }

  // Bird
  ctx.save();
  const rotation = Math.max(-0.5, Math.min(0.9, bird.velocity / 12));
  ctx.translate(BIRD_X, bird.y);
  ctx.rotate(rotation);
  ctx.fillStyle = mode === "ai" ? "#ff4d9e" : "#ffc857";
  ctx.beginPath();
  ctx.arc(0, 0, BIRD_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  // little eye + beak so it reads as a bird, not just a dot
  ctx.fillStyle = "#0a0e17";
  ctx.beginPath();
  ctx.arc(5, -4, 2.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#ff8a3d";
  ctx.beginPath();
  ctx.moveTo(BIRD_RADIUS - 2, -2);
  ctx.lineTo(BIRD_RADIUS + 8, 2);
  ctx.lineTo(BIRD_RADIUS - 2, 6);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ---------------- Game loop ----------------
function loop() {
  update();
  draw();
  if (alive) {
    animationId = requestAnimationFrame(loop);
  }
}

function startLoop() {
  cancelAnimationFrame(animationId);
  animationId = requestAnimationFrame(loop);
}

// ---------------- Game over handling ----------------
function onGameOver() {
  if (score > bestScore) {
    bestScore = score;
    localStorage.setItem("flappyBestScore", bestScore);
    bestScoreValue.textContent = bestScore;
  }

  if (mode === "manual") {
    finalScoreText.textContent = `Score: ${score}`;
    gameOverOverlay.classList.remove("hidden");
  } else if (mode === "ai") {
    // Automatically train the next generation and play it
    trainNextGeneration();
  }
}

// ---------------- Manual controls ----------------
function handleFlapInput() {
  if (mode !== "manual") return;
  if (!started) return;
  if (!alive) return;
  flap();
}

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    handleFlapInput();
  }
});
canvas.addEventListener("mousedown", handleFlapInput);
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  handleFlapInput();
});

// ---------------- Buttons: Start / Restart ----------------
function beginManualGame() {
  mode = "manual";
  modePill.textContent = "MANUAL";
  modePill.classList.remove("ai-active");
  aiModeBtn.classList.remove("active");
  setStatus("idle", "Idle — click \"AI Mode\" to start training");

  started = true;
  startOverlay.classList.add("hidden");
  gameOverOverlay.classList.add("hidden");
  resetGame();
  startLoop();
}

startBtn.addEventListener("click", beginManualGame);
restartBtn.addEventListener("click", () => {
  if (mode === "ai") return; // restart button only applies to manual play
  beginManualGame();
});
restartBtnOverlay.addEventListener("click", beginManualGame);

// ---------------- AI Mode ----------------
function setStatus(kind, text) {
  statusDot.className = "status-dot" + (kind === "training" ? " training" : kind === "playing" ? " playing" : "");
  statusText.textContent = text;
}

async function trainNextGeneration() {
  if (aiTrainingBusy) return;
  aiTrainingBusy = true;
  setStatus("training", `Training generation ${parseInt(generationValue.textContent) + 1}… evaluating population`);

  try {
    const res = await fetch("/api/train_generation", { method: "POST" });
    const data = await res.json();

    generationValue.textContent = data.generation;
    aiBestValue.textContent = data.best_score_ever;
    aiWeights = data.weights;

    setStatus("playing", `Generation ${data.generation}: best of gen scored ${data.best_score_this_gen}, avg ${data.avg_score_this_gen}`);
  } catch (err) {
    setStatus("idle", "Could not reach the AI server — is Flask running?");
    aiTrainingBusy = false;
    return;
  }

  aiTrainingBusy = false;

  // Only keep playing automatically if we're still in AI mode
  if (mode === "ai") {
    started = true;
    startOverlay.classList.add("hidden");
    gameOverOverlay.classList.add("hidden");
    resetGame();
    startLoop();
  }
}

function startAIMode() {
  mode = "ai";
  modePill.textContent = "AI LEARNING";
  modePill.classList.add("ai-active");
  aiModeBtn.classList.add("active");
  aiModeBtn.textContent = "■ Stop AI";
  trainNextGeneration();
}

function stopAIMode() {
  mode = "manual";
  modePill.textContent = "MANUAL";
  modePill.classList.remove("ai-active");
  aiModeBtn.classList.remove("active");
  aiModeBtn.textContent = "⚡ AI Mode";
  setStatus("idle", "AI paused — click \"AI Mode\" to resume training");
  cancelAnimationFrame(animationId);
  started = false;
  startOverlay.classList.remove("hidden");
  gameOverOverlay.classList.add("hidden");
}

aiModeBtn.addEventListener("click", () => {
  if (mode === "ai") {
    stopAIMode();
  } else {
    startAIMode();
  }
});

resetAiBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  generationValue.textContent = "0";
  aiBestValue.textContent = "0";
  aiWeights = null;
  setStatus("idle", "AI reset — starting fresh from generation 0");
});

// ---------------- Initial paint ----------------
resetGame();
draw();
