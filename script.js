// RL Blocks MVP: editable grid-world with basic Q-learning.
// This file uses plain JavaScript so the site works directly on GitHub Pages.

const gridSize = 5;
const initialEpsilon = 0.25;

const actions = [
  { name: "up", rowChange: -1, colChange: 0, arrow: "^" },
  { name: "right", rowChange: 0, colChange: 1, arrow: ">" },
  { name: "down", rowChange: 1, colChange: 0, arrow: "v" },
  { name: "left", rowChange: 0, colChange: -1, arrow: "<" }
];

const rewards = {
  goal: 10,
  invalidMove: -2,
  normalMove: -0.1
};

// These are the main Q-learning settings.
// alpha: how strongly new experience changes old knowledge.
// gamma: how much the robot cares about future rewards.
// epsilon: how often the robot explores instead of choosing its current best move.
const alpha = 0.2;
const gamma = 0.9;
let epsilon = initialEpsilon;

let qTable = {};
let robotStartState = null;
let goalState = null;
let agentState = null;
let walls = new Set();
let visitedStates = new Set();
let episode = 0;
let steps = 0;
let totalReward = 0;
let selectedTool = "robot";
let isAnimating = false;

const gridElement = document.querySelector("#grid");
const qTableBody = document.querySelector("#qTableBody");
const episodeStat = document.querySelector("#episodeStat");
const stepsStat = document.querySelector("#stepsStat");
const rewardStat = document.querySelector("#rewardStat");
const statusMessage = document.querySelector("#statusMessage");
const randomMoveBtn = document.querySelector("#randomMoveBtn");
const runEpisodeBtn = document.querySelector("#runEpisodeBtn");
const trainBtn = document.querySelector("#trainBtn");
const resetBtn = document.querySelector("#resetBtn");
const toolButtons = document.querySelectorAll(".tool-button");

toolButtons.forEach((button) => {
  button.addEventListener("click", () => selectTool(button.dataset.tool));
  button.addEventListener("dragstart", (event) => {
    selectTool(button.dataset.tool);
    event.dataTransfer.setData("text/plain", button.dataset.tool);
  });
});

randomMoveBtn.addEventListener("click", moveRandomly);
runEpisodeBtn.addEventListener("click", runBestEpisode);
trainBtn.addEventListener("click", trainAgent);
resetBtn.addEventListener("click", resetLearning);

render();
setStatus("Place both the robot and goal before training.", "warning");

function stateKey(state) {
  return `${state.row},${state.col}`;
}

function stateLabel(state) {
  return `row ${state.row + 1}, col ${state.col + 1}`;
}

function sameState(first, second) {
  return Boolean(
    first &&
    second &&
    first.row === second.row &&
    first.col === second.col
  );
}

function isWall(state) {
  return walls.has(stateKey(state));
}

function isInsideGrid(state) {
  return (
    state.row >= 0 &&
    state.row < gridSize &&
    state.col >= 0 &&
    state.col < gridSize
  );
}

function isTerminalState(state) {
  return sameState(state, goalState);
}

function getAllStates() {
  const states = [];

  for (let row = 0; row < gridSize; row += 1) {
    for (let col = 0; col < gridSize; col += 1) {
      states.push({ row, col });
    }
  }

  return states;
}

function getQValues(state) {
  const key = stateKey(state);

  if (qTable[key]) {
    return qTable[key];
  }

  return { up: 0, right: 0, down: 0, left: 0 };
}

function ensureQValues(state) {
  const key = stateKey(state);

  if (!qTable[key]) {
    qTable[key] = { up: 0, right: 0, down: 0, left: 0 };
  }

  return qTable[key];
}

function chooseRandomAction() {
  const randomIndex = Math.floor(Math.random() * actions.length);
  return actions[randomIndex];
}

function chooseBestAction(state, useRandomTieBreak = true) {
  const qValues = getQValues(state);
  const highestValue = Math.max(...actions.map((action) => qValues[action.name]));
  const bestActions = actions.filter((action) => qValues[action.name] === highestValue);

  if (useRandomTieBreak) {
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  return bestActions[0];
}

function chooseEpsilonGreedyAction(state) {
  if (Math.random() < epsilon) {
    return chooseRandomAction();
  }

  return chooseBestAction(state);
}

function getMoveResult(state, action) {
  const nextState = {
    row: state.row + action.rowChange,
    col: state.col + action.colChange
  };

  if (!isInsideGrid(nextState) || isWall(nextState)) {
    return {
      nextState: { ...state },
      reward: rewards.invalidMove,
      hitObstacle: true
    };
  }

  if (isTerminalState(nextState)) {
    return {
      nextState,
      reward: rewards.goal,
      reachedGoal: true
    };
  }

  return {
    nextState,
    reward: rewards.normalMove
  };
}

function learnFromMove(state, action, reward, nextState) {
  const qValues = ensureQValues(state);
  const nextQValues = ensureQValues(nextState);
  const oldValue = qValues[action.name];
  const bestFutureValue = Math.max(...Object.values(nextQValues));

  // Q-learning update rule:
  // new value = old value + alpha * (reward + future value - old value)
  qValues[action.name] =
    oldValue + alpha * (reward + gamma * bestFutureValue - oldValue);
}

function applyMove(action, shouldLearn) {
  const oldState = { ...agentState };
  const result = getMoveResult(agentState, action);

  if (shouldLearn) {
    learnFromMove(oldState, action, result.reward, result.nextState);
  }

  agentState = result.nextState;
  steps += 1;
  totalReward += result.reward;
  visitedStates.add(stateKey(agentState));

  if (result.reachedGoal) {
    setStatus(`Goal reached! Reward: +${rewards.goal}`, "success");
  } else if (result.hitObstacle) {
    setStatus(`That move hit a wall or edge. Reward: ${rewards.invalidMove}`, "warning");
  } else {
    setStatus(`Moved ${action.name}. Reward: ${rewards.normalMove}`, "");
  }

  render();
  return result;
}

function moveRandomly() {
  if (isAnimating || !isGridReady()) {
    return;
  }

  if (isTerminalState(agentState)) {
    startNewEpisode();
  }

  const action = chooseRandomAction();
  applyMove(action, true);
}

function startNewEpisode() {
  episode += 1;
  steps = 0;
  totalReward = 0;
  agentState = { ...robotStartState };
  visitedStates = new Set([stateKey(agentState)]);
}

async function runBestEpisode() {
  if (isAnimating || !isGridReady()) {
    return;
  }

  isAnimating = true;
  updateControls();
  startNewEpisode();
  setStatus("Following the best learned path...", "");
  render();

  for (let i = 0; i < 30; i += 1) {
    if (isTerminalState(agentState)) {
      break;
    }

    const bestAction = chooseBestAction(agentState);
    const result = applyMove(bestAction, false);
    await wait(250);

    if (result.reachedGoal) {
      break;
    }
  }

  if (!isTerminalState(agentState)) {
    setStatus("The robot did not reach the goal yet. Try training more.", "warning");
  }

  isAnimating = false;
  updateControls();
}

async function trainAgent() {
  if (isAnimating || !isGridReady()) {
    return;
  }

  isAnimating = true;
  updateControls();
  setStatus("Training for 400 practice episodes...", "");

  const trainingEpisodes = 400;
  const maxStepsPerEpisode = 60;

  for (let i = 0; i < trainingEpisodes; i += 1) {
    let trainingState = { ...robotStartState };

    for (let step = 0; step < maxStepsPerEpisode; step += 1) {
      const action = chooseEpsilonGreedyAction(trainingState);
      const result = getMoveResult(trainingState, action);

      learnFromMove(trainingState, action, result.reward, result.nextState);
      trainingState = result.nextState;

      if (result.reachedGoal) {
        break;
      }
    }
  }

  episode += trainingEpisodes;
  resetEpisodeStatsOnly();
  epsilon = Math.max(0.05, epsilon * 0.9);
  setStatus("Training complete. Press Run Episode to watch the learned path.", "success");
  render();

  isAnimating = false;
  updateControls();
}

function selectTool(tool) {
  selectedTool = tool;

  toolButtons.forEach((button) => {
    button.classList.toggle("selected", button.dataset.tool === tool);
  });
}

function placeToolOnCell(tool, state) {
  if (isAnimating) {
    return;
  }

  if (tool === "robot") {
    if (isWall(state) || sameState(state, goalState)) {
      setStatus("Robot cannot be placed on a wall or the goal.", "warning");
      return;
    }

    robotStartState = { ...state };
    agentState = { ...state };
    clearLearningAfterGridChange();
    return;
  }

  if (tool === "goal") {
    if (isWall(state) || sameState(state, robotStartState)) {
      setStatus("Goal cannot be placed on a wall or the robot.", "warning");
      return;
    }

    goalState = { ...state };
    clearLearningAfterGridChange();
    return;
  }

  if (tool === "wall") {
    if (sameState(state, robotStartState) || sameState(state, goalState)) {
      setStatus("Walls cannot be placed on the robot or goal.", "warning");
      return;
    }

    const key = stateKey(state);

    if (walls.has(key)) {
      walls.delete(key);
    } else {
      walls.add(key);
    }

    clearLearningAfterGridChange();
  }
}

function clearLearningAfterGridChange() {
  qTable = {};
  episode = 0;
  epsilon = initialEpsilon;
  resetEpisodeStatsOnly();
  setStatus("Grid changed. Train the agent again.", "");
  render();
}

function resetLearning() {
  qTable = {};
  episode = 0;
  epsilon = initialEpsilon;
  resetEpisodeStatsOnly();
  setStatus("Learning reset. The grid stayed the same.", "");
  render();
}

function resetEpisodeStatsOnly() {
  steps = 0;
  totalReward = 0;

  if (robotStartState) {
    agentState = { ...robotStartState };
    visitedStates = new Set([stateKey(agentState)]);
  } else {
    agentState = null;
    visitedStates = new Set();
  }
}

function isGridReady() {
  const validation = getGridValidation();
  return validation.ready;
}

function getGridValidation() {
  if (!robotStartState || !goalState) {
    return {
      ready: false,
      message: "Place both the robot and goal before training."
    };
  }

  if (!hasPathFromRobotToGoal()) {
    return {
      ready: false,
      message: "No possible path exists. Move or erase walls so the robot can reach the goal."
    };
  }

  return {
    ready: true,
    message: ""
  };
}

function hasPathFromRobotToGoal() {
  const queue = [{ ...robotStartState }];
  const seen = new Set([stateKey(robotStartState)]);

  while (queue.length > 0) {
    const current = queue.shift();

    if (sameState(current, goalState)) {
      return true;
    }

    actions.forEach((action) => {
      const nextState = {
        row: current.row + action.rowChange,
        col: current.col + action.colChange
      };
      const nextKey = stateKey(nextState);

      if (
        isInsideGrid(nextState) &&
        !isWall(nextState) &&
        !seen.has(nextKey)
      ) {
        seen.add(nextKey);
        queue.push(nextState);
      }
    });
  }

  return false;
}

function render() {
  gridElement.innerHTML = "";

  getAllStates().forEach((cellState) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.row = cellState.row;
    cell.dataset.col = cellState.col;

    cell.addEventListener("click", () => placeToolOnCell(selectedTool, cellState));
    cell.addEventListener("dragover", (event) => event.preventDefault());
    cell.addEventListener("dragenter", () => cell.classList.add("drop-target"));
    cell.addEventListener("dragleave", () => cell.classList.remove("drop-target"));
    cell.addEventListener("drop", (event) => {
      event.preventDefault();
      cell.classList.remove("drop-target");
      placeToolOnCell(event.dataTransfer.getData("text/plain") || selectedTool, cellState);
    });

    if (visitedStates.has(stateKey(cellState))) {
      cell.classList.add("visited");
    }

    if (isWall(cellState)) {
      cell.classList.add("wall");
    }

    if (sameState(cellState, goalState)) {
      cell.classList.add("goal");
      cell.textContent = "G";
    }

    if (!isWall(cellState) && !isTerminalState(cellState)) {
      const qValues = getQValues(cellState);
      const hasLearnedValue = Object.values(qValues).some((value) => value !== 0);

      if (hasLearnedValue) {
        const bestAction = chooseBestAction(cellState, false);
        const arrow = document.createElement("span");
        arrow.className = "policy-arrow";
        arrow.textContent = bestAction.arrow;
        cell.appendChild(arrow);
      }
    }

    if (sameState(cellState, agentState)) {
      const agent = document.createElement("span");
      agent.className = "agent-token";
      agent.textContent = "R";
      cell.appendChild(agent);
    }

    gridElement.appendChild(cell);
  });

  episodeStat.textContent = episode;
  stepsStat.textContent = steps;
  rewardStat.textContent = totalReward.toFixed(2);
  renderQTable();
  updateControls();
}

function renderQTable() {
  qTableBody.innerHTML = "";

  getAllStates().forEach((state) => {
    const row = document.createElement("tr");
    const qValues = getQValues(state);

    if (isWall(state)) {
      row.classList.add("wall-row");
    }

    row.appendChild(makeTableCell(getStateDescription(state)));

    actions.forEach((action) => {
      row.appendChild(makeTableCell(qValues[action.name].toFixed(2)));
    });

    qTableBody.appendChild(row);
  });
}

function getStateDescription(state) {
  const parts = [stateLabel(state)];

  if (sameState(state, robotStartState)) {
    parts.push("robot start");
  }

  if (sameState(state, goalState)) {
    parts.push("goal");
  }

  if (isWall(state)) {
    parts.push("wall");
  }

  return parts.join(" - ");
}

function makeTableCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function updateControls() {
  const validation = getGridValidation();
  const disabledBecauseInvalid = !validation.ready;

  randomMoveBtn.disabled = isAnimating || disabledBecauseInvalid;
  runEpisodeBtn.disabled = isAnimating || disabledBecauseInvalid;
  trainBtn.disabled = isAnimating || disabledBecauseInvalid;
  resetBtn.disabled = isAnimating;

  if (!isAnimating && disabledBecauseInvalid) {
    setStatus(validation.message, "warning");
  }
}

function setStatus(message, type) {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

function wait(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}
