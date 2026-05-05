// RL Blocks MVP: a tiny grid-world with basic Q-learning.
// The code is intentionally plain JavaScript so it can run on GitHub Pages.

const gridSize = 5;
const startState = { row: 0, col: 0 };
const goalState = { row: 4, col: 4 };
const walls = [
  { row: 1, col: 1 },
  { row: 1, col: 3 },
  { row: 2, col: 3 },
  { row: 3, col: 1 }
];

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
let epsilon = 0.25;

let qTable = {};
let agentState = { ...startState };
let visitedStates = new Set([stateKey(agentState)]);
let episode = 0;
let steps = 0;
let totalReward = 0;
let isAnimating = false;

const gridElement = document.querySelector("#grid");
const episodeStat = document.querySelector("#episodeStat");
const stepsStat = document.querySelector("#stepsStat");
const rewardStat = document.querySelector("#rewardStat");
const statusMessage = document.querySelector("#statusMessage");
const randomMoveBtn = document.querySelector("#randomMoveBtn");
const runEpisodeBtn = document.querySelector("#runEpisodeBtn");
const trainBtn = document.querySelector("#trainBtn");
const resetBtn = document.querySelector("#resetBtn");

randomMoveBtn.addEventListener("click", moveRandomly);
runEpisodeBtn.addEventListener("click", runBestEpisode);
trainBtn.addEventListener("click", trainAgent);
resetBtn.addEventListener("click", resetSimulation);

render();

function stateKey(state) {
  return `${state.row},${state.col}`;
}

function sameState(first, second) {
  return first.row === second.row && first.col === second.col;
}

function isWall(state) {
  return walls.some((wall) => sameState(wall, state));
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

function ensureQValues(state) {
  const key = stateKey(state);

  if (!qTable[key]) {
    qTable[key] = {};
    actions.forEach((action) => {
      qTable[key][action.name] = 0;
    });
  }

  return qTable[key];
}

function chooseRandomAction() {
  const randomIndex = Math.floor(Math.random() * actions.length);
  return actions[randomIndex];
}

function chooseBestAction(state) {
  const qValues = ensureQValues(state);
  const highestValue = Math.max(...actions.map((action) => qValues[action.name]));
  const bestActions = actions.filter((action) => qValues[action.name] === highestValue);

  return bestActions[Math.floor(Math.random() * bestActions.length)];
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
  if (isAnimating) {
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
  agentState = { ...startState };
  visitedStates = new Set([stateKey(agentState)]);
}

async function runBestEpisode() {
  if (isAnimating) {
    return;
  }

  isAnimating = true;
  setButtonsDisabled(true);
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
  setButtonsDisabled(false);
}

async function trainAgent() {
  if (isAnimating) {
    return;
  }

  isAnimating = true;
  setButtonsDisabled(true);
  setStatus("Training for 400 practice episodes...", "");

  const trainingEpisodes = 400;
  const maxStepsPerEpisode = 60;

  for (let i = 0; i < trainingEpisodes; i += 1) {
    let trainingState = { ...startState };

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
  setButtonsDisabled(false);
}

function resetSimulation() {
  qTable = {};
  episode = 0;
  epsilon = 0.25;
  agentState = { ...startState };
  visitedStates = new Set([stateKey(agentState)]);
  resetEpisodeStatsOnly();
  setStatus("Reset complete. The robot is ready to learn again.", "");
  render();
}

function resetEpisodeStatsOnly() {
  steps = 0;
  totalReward = 0;
  agentState = { ...startState };
  visitedStates = new Set([stateKey(agentState)]);
}

function render() {
  gridElement.innerHTML = "";

  for (let row = 0; row < gridSize; row += 1) {
    for (let col = 0; col < gridSize; col += 1) {
      const cellState = { row, col };
      const cell = document.createElement("div");
      cell.className = "cell";

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
        const bestAction = chooseBestAction(cellState);
        const hasLearnedValue = Object.values(ensureQValues(cellState)).some((value) => value !== 0);

        if (hasLearnedValue) {
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
    }
  }

  episodeStat.textContent = episode;
  stepsStat.textContent = steps;
  rewardStat.textContent = totalReward.toFixed(2);
}

function setStatus(message, type) {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

function setButtonsDisabled(disabled) {
  randomMoveBtn.disabled = disabled;
  runEpisodeBtn.disabled = disabled;
  trainBtn.disabled = disabled;
  resetBtn.disabled = disabled;
}

function wait(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}
