# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

st.title("🕹️ Fun Hub: Five Mini-Games")
st.markdown("Select a tab below to try a different interactive game. High scores are saved locally in your browser!")

# --- 1. Snake Game HTML/JS Content ---

SNAKE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0d1117; }
        #snakeCanvas {
            background-color: #2c3e50;
            border: 4px solid #27ae60;
            box-shadow: 0 0 15px rgba(39, 174, 96, 0.7);
            display: block;
            margin: 0 auto;
            /* Fixed size for a crisp 13x13 grid */
            width: 520px; 
            height: 520px;
            max-width: 95vw;
            border-radius: 12px;
        }
        .snake-message-box {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 100;
            min-width: 280px;
        }
    </style>
</head>
<body class="p-4 md:p-8">

    <div class="max-w-4xl mx-auto">
        <h2 class="text-3xl font-bold text-white mb-4 text-center">🐍 Classic Snake</h2>
        <p class="text-gray-400 text-center mb-6">
            Use the **Arrow Keys** to guide the snake to the red food!
        </p>

        <div class="relative flex flex-col items-center">
            <!-- Canvas is set to a specific size for 13x13 grid of 40px tiles -->
            <canvas id="snakeCanvas" width="520" height="520"></canvas>

            <!-- Message Box for Start/Game Over -->
            <div id="snakeMessageBox" class="snake-message-box bg-gray-800 p-8 rounded-xl shadow-2xl border-2 border-yellow-400 text-center transition duration-300">
                <h3 id="snakeMessageText" class="text-3xl font-bold text-white mb-4">Game Over!</h3>
                <div class="text-xl text-gray-300 mb-6">Score: <span id="snakeFinalScore" class="font-extrabold text-green-400">0</span></div>
                <button id="snakeStartButton" class="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-md">
                    Start Game (Press Enter)
                </button>
            </div>

            <!-- Score HUD -->
            <div class="mt-4 w-full flex justify-between text-white text-xl font-mono p-4 bg-gray-700 rounded-lg shadow-inner max-w-[520px]">
                <div>Score: <span id="snakeScoreDisplay" class="text-green-400 font-bold">0</span></div>
                <div>High Score: <span id="snakeHighScoreDisplay" class="text-yellow-400 font-bold">0</span></div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('snakeCanvas');
            const ctx = canvas.getContext('2d');
            const scoreDisplay = document.getElementById('snakeScoreDisplay');
            const highScoreDisplay = document.getElementById('snakeHighScoreDisplay');
            const messageBox = document.getElementById('snakeMessageBox');
            const messageText = document.getElementById('snakeMessageText');
            const finalScore = document.getElementById('snakeFinalScore');
            const startButton = document.getElementById('snakeStartButton');

            // Game constants
            const GRID_SIZE = 13; 
            const CANVAS_SIZE = canvas.width;
            const TILE_SIZE = CANVAS_SIZE / GRID_SIZE; 

            // Game state
            let snake;
            let food;
            let dx, dy;
            let score;
            let highScore = 0;
            let gameLoopInterval;
            let gameSpeed = 200; // Slower start speed (was 180)

            function loadHighScore() {
                const storedScore = localStorage.getItem('snakeHighScore');
                highScore = storedScore ? parseInt(storedScore, 10) : 0;
            }

            function saveHighScore() {
                if (score > highScore) {
                    highScore = score;
                    localStorage.setItem('snakeHighScore', highScore);
                }
            }

            function updateHUD() {
                scoreDisplay.textContent = score;
                highScoreDisplay.textContent = highScore;
            }

            function resetGame() {
                loadHighScore();
                snake = [{x: 6, y: 6}, {x: 5, y: 6}, {x: 4, y: 6}]; 
                dx = 1; dy = 0;
                score = 0;
                gameSpeed = 200; // Reset speed
                updateHUD();
                spawnFood();
            }

            function drawRectangle(x, y, color) {
                ctx.fillStyle = color;
                ctx.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            }

            function drawSnake() {
                snake.forEach((segment, index) => {
                    const color = index === 0 ? '#2ecc71' : '#27ae60'; 
                    drawRectangle(segment.x, segment.y, color);
                });
            }

            function spawnFood() {
                food = {
                    x: Math.floor(Math.random() * GRID_SIZE),
                    y: Math.floor(Math.random() * GRID_SIZE)
                };

                for (const segment of snake) {
                    if (segment.x === food.x && segment.y === food.y) {
                        spawnFood();
                        return;
                    }
                }
            }

            function moveSnake() {
                const head = {x: snake[0].x + dx, y: snake[0].y + dy};
                snake.unshift(head);

                if (head.x === food.x && head.y === food.y) {
                    score += 10;
                    updateHUD();
                    
                    // Slightly increase speed
                    gameSpeed = Math.max(100, gameSpeed - 3); 
                    clearInterval(gameLoopInterval);
                    gameLoopInterval = setInterval(gameLoop, gameSpeed);

                    spawnFood();
                } else {
                    snake.pop();
                }
            }
            
            function checkCollision() {
                const head = snake[0];

                // Wall collision (using GRID_SIZE = 13)
                if (head.x < 0 || head.x >= GRID_SIZE || head.y < 0 || head.y >= GRID_SIZE) {
                    return true;
                }

                // Self-collision
                for (let i = 1; i < snake.length; i++) {
                    if (head.x === snake[i].x && head.y === snake[i].y) {
                        return true;
                    }
                }
                return false;
            }

            function gameLoop() {
                if (checkCollision()) {
                    gameOver();
                    return;
                }

                // Clear canvas
                ctx.fillStyle = '#2c3e50';
                ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

                moveSnake();
                
                // Draw food
                ctx.fillStyle = '#e74c3c';
                ctx.shadowColor = '#c0392b';
                ctx.shadowBlur = 10;
                ctx.fillRect(food.x * TILE_SIZE, food.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                ctx.shadowBlur = 0;
                
                drawSnake();
            }

            function handleDirectionChange(newDx, newDy) {
                if ((dx === -newDx && dx !== 0) || (dy === -newDy && dy !== 0)) {
                    return;
                }
                dx = newDx;
                dy = newDy;
            }

            function handleInput(e) {
                if (messageBox.style.display !== 'none' && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    startGame();
                    return;
                }
                
                switch (e.key) {
                    case 'ArrowUp':
                    case 'w':
                        e.preventDefault();
                        handleDirectionChange(0, -1);
                        break;
                    case 'ArrowDown':
                    case 's':
                        e.preventDefault();
                        handleDirectionChange(0, 1);
                        break;
                    case 'ArrowLeft':
                    case 'a':
                        e.preventDefault();
                        handleDirectionChange(-1, 0);
                        break;
                    case 'ArrowRight':
                    case 'd':
                        e.preventDefault();
                        handleDirectionChange(1, 0);
                        break;
                }
            }

            function gameOver() {
                clearInterval(gameLoopInterval);
                saveHighScore();
                updateHUD();
                messageText.textContent = "Game Over!";
                finalScore.textContent = score;
                startButton.textContent = "Play Again";
                messageBox.style.display = 'block';
            }

            function startGame() {
                messageBox.style.display = 'none';
                resetGame();
                gameLoopInterval = setInterval(gameLoop, gameSpeed);
            }

            // --- Initial Setup ---
            document.addEventListener('keydown', handleInput);
            startButton.addEventListener('click', startGame);

            loadHighScore();
            gameOver(); 
            messageText.textContent = "Welcome to Snake!";
            finalScore.textContent = 0;
        });
    </script>
</body>
</html>
"""

# --- 2. Obstacle Runner Game HTML/JS Content---

OBSTACLE_RUNNER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0d1117; }
        #gameCanvas {
            background-color: #161b22;
            border: 4px solid #4a90e2;
            box-shadow: 0 0 15px rgba(74, 144, 226, 0.7);
            display: block;
            margin: 0 auto;
            max-width: 95vw;
            border-radius: 12px;
        }
        .message-box {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 100;
            min-width: 280px;
        }
    </style>
</head>
<body class="p-4 md:p-8">

    <div class="max-w-4xl mx-auto">
        <h2 class="text-3xl font-bold text-white mb-4 text-center">Avoid the Obstacles</h2>
        <p class="text-gray-400 text-center mb-6">
            Use the **Arrow Keys** to move. Difficulty increases with score!
        </p>

        <div class="relative flex flex-col items-center">
            <canvas id="gameCanvas" width="800" height="450"></canvas>

            <div id="messageBox" class="message-box bg-gray-800 p-8 rounded-xl shadow-2xl border-2 border-green-400 text-center transition duration-300">
                <h3 id="messageText" class="text-3xl font-bold text-white mb-4">Game Over!</h3>
                <div class="text-xl text-gray-300 mb-6">Score: <span id="finalScore" class="font-extrabold text-yellow-400">0</span></div>
                <button id="startButton" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-md">
                    Start Game (Press Enter)
                </button>
            </div>

            <div class="mt-4 w-full flex justify-between text-white text-xl font-mono p-4 bg-gray-700 rounded-lg shadow-inner max-w-[800px]">
                <div>Score: <span id="scoreDisplay" class="text-yellow-400 font-bold">0</span></div>
                <div>Lives: <span id="livesDisplay" class="text-red-400 font-bold">3</span></div>
                <div>High Score: <span id="highScoreDisplay" class="text-green-400 font-bold">0</span></div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('gameCanvas');
            const ctx = canvas.getContext('2d');
            const scoreDisplay = document.getElementById('scoreDisplay');
            const livesDisplay = document.getElementById('livesDisplay');
            const highScoreDisplay = document.getElementById('highScoreDisplay');
            const messageBox = document.getElementById('messageBox');
            const messageText = document.getElementById('messageText');
            const finalScore = document.getElementById('finalScore');
            const startButton = document.getElementById('startButton');

            let gameRunning = false;
            let score = 0;
            let highScore = 0;
            let lives = 3;
            let difficultyLevel = 1;
            const BASE_SPEED = 3; 
            const BASE_SPAWN_RATE = 1000; 
            const PLAYER_SPEED = 7;

            let player = {
                x: canvas.width / 2,
                y: canvas.height - 50,
                width: 25,
                height: 25,
                speed: PLAYER_SPEED,
                dx: 0,
                dy: 0,
                color: '#4a90e2'
            };
            let obstacles = [];
            let obstacleSpeed = BASE_SPEED;
            let obstacleSpawnRate = BASE_SPAWN_RATE;
            let lastSpawnTime = 0;
            let animationFrameId;
            
            function loadHighScore() {
                const storedScore = localStorage.getItem('runnerHighScore');
                highScore = storedScore ? parseInt(storedScore, 10) : 0;
            }

            function saveHighScore() {
                if (score > highScore) {
                    highScore = score;
                    localStorage.setItem('runnerHighScore', highScore);
                }
            }

            const keys = {};
            
            document.addEventListener('keydown', (e) => {
                if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key) && gameRunning) {
                    e.preventDefault();
                }

                keys[e.key] = true;
                if (!gameRunning && (e.key === 'Enter' || e.key === ' ')) {
                    if (messageBox.style.display !== 'none') {
                        startGame();
                    }
                }
            });

            document.addEventListener('keyup', (e) => {
                keys[e.key] = false;
            });
            
            startButton.addEventListener('click', startGame);

            function drawPlayer() {
                ctx.fillStyle = player.color;
                ctx.shadowColor = player.color;
                ctx.shadowBlur = 10;
                ctx.fillRect(player.x, player.y, player.width, player.height);
                ctx.shadowBlur = 0;
            }

            function updatePlayer() {
                player.dx = 0;
                player.dy = 0;

                if (keys['ArrowLeft'] || keys['a']) player.dx = -player.speed;
                if (keys['ArrowRight'] || keys['d']) player.dx = player.speed;
                if (keys['ArrowUp'] || keys['w']) player.dy = -player.speed;
                if (keys['ArrowDown'] || keys['s']) player.dy = player.speed;

                player.x += player.dx;
                player.y += player.dy;

                player.x = Math.max(0, Math.min(player.x, canvas.width - player.width));
                player.y = Math.max(0, Math.min(player.y, canvas.height - player.height));
            }

            function spawnObstacle() {
                const minSize = 10;
                const maxSize = 40 + (difficultyLevel * 5); 
                const size = Math.random() * (maxSize - minSize) + minSize; 
                const x = Math.random() * (canvas.width - size);
                
                obstacles.push({
                    x: x,
                    y: -size,
                    width: size,
                    height: size,
                    speed: obstacleSpeed * (Math.random() * 0.5 + 0.8),
                    color: '#e74c3c'
                });
            }

            function updateObstacles() {
                for (let i = 0; i < obstacles.length; i++) {
                    const obs = obstacles[i];
                    obs.y += obs.speed;

                    if (
                        player.x < obs.x + obs.width &&
                        player.x + player.width > obs.x &&
                        player.y < obs.y + obs.height &&
                        player.y + player.height > obs.y
                    ) {
                        handleCollision();
                        obstacles.splice(i, 1);
                        i--;
                        continue;
                    }

                    if (obs.y > canvas.height) {
                        score += 10;
                        checkDifficultyIncrease();
                        updateHUD();
                        obstacles.splice(i, 1);
                        i--;
                    }
                }
            }

            function drawObstacles() {
                ctx.fillStyle = '#e74c3c';
                ctx.shadowColor = '#c0392b';
                ctx.shadowBlur = 8;
                for (const obs of obstacles) {
                    ctx.fillRect(obs.x, obs.y, obs.width, obs.height);
                }
                ctx.shadowBlur = 0;
            }

            function handleCollision() {
                lives -= 1;
                updateHUD();
                
                player.color = '#ff0000';
                setTimeout(() => { player.color = '#4a90e2'; }, 100);

                if (lives <= 0) {
                    gameOver();
                }
            }
            
            function checkDifficultyIncrease() {
                // MODIFIED: Lowered score threshold and increased multipliers for rapid difficulty scaling
                const newDifficultyLevel = Math.floor(score / 250) + 1; // Level up every 250 points
                
                if (newDifficultyLevel > difficultyLevel) {
                    difficultyLevel = newDifficultyLevel;
                    
                    // Increased speed multiplier (now 1.0)
                    obstacleSpeed = BASE_SPEED + (difficultyLevel * 1.0); 
                    
                    // Faster reduction in spawn rate (now 200)
                    const newSpawnRate = BASE_SPAWN_RATE - (difficultyLevel * 200); 
                    obstacleSpawnRate = Math.max(100, newSpawnRate); // Minimum spawn rate is now 100ms
                }
            }

            function updateHUD() {
                scoreDisplay.textContent = score;
                livesDisplay.textContent = lives;
                highScoreDisplay.textContent = highScore;
            }

            function resetGame() {
                loadHighScore();
                score = 0;
                lives = 3;
                difficultyLevel = 1;
                obstacleSpeed = BASE_SPEED;
                obstacleSpawnRate = BASE_SPAWN_RATE;
                player.x = canvas.width / 2;
                player.y = canvas.height - 50;
                obstacles = [];
                lastSpawnTime = 0;
                updateHUD();
                
                player.color = '#4a90e2'; 
                
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
            }

            function gameOver() {
                gameRunning = false;
                cancelAnimationFrame(animationFrameId);
                saveHighScore();

                messageText.textContent = "Game Over!";
                finalScore.textContent = score;
                startButton.textContent = "Play Again";
                messageBox.style.display = 'block';
                updateHUD();
            }

            function startGame() {
                resetGame();
                gameRunning = true;
                messageBox.style.display = 'none';
                gameLoop();
            }

            function gameLoop(timestamp) {
                if (!gameRunning) return;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                updatePlayer();
                updateObstacles();

                if (timestamp - lastSpawnTime > obstacleSpawnRate) {
                    spawnObstacle();
                    lastSpawnTime = timestamp;
                }

                drawObstacles();
                drawPlayer();
                
                animationFrameId = requestAnimationFrame(gameLoop);
            }

            resetGame();
            gameOver();

            function resizeCanvas() {
                const containerWidth = canvas.parentElement.clientWidth;
                const newWidth = Math.min(800, containerWidth);
                const newHeight = newWidth * (450 / 800); 
                
                canvas.width = newWidth;
                canvas.height = newHeight;

                if (!gameRunning) {
                    player.x = canvas.width / 2;
                    player.y = canvas.height - 50;
                }
            }

            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();
        });
    </script>
</body>
</html>
"""

# --- 3. Color Clicker Game HTML/JS Content ---

COLOR_CLICKER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .game-container {
            min-height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background-color: #0d1117;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        #colorBox {
            width: 250px;
            height: 250px;
            border-radius: 12px;
            margin: 20px 0;
            cursor: pointer;
            transition: background-color 0.1s;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="game-container max-w-lg mx-auto text-white">
        <h2 class="text-3xl font-bold mb-2">Color Clicker (Reaction Test)</h2>
        <p class="text-gray-400 mb-4 text-center">Click the square immediately when it turns **GREEN**.</p>
        
        <div id="colorBox" class="bg-gray-700 flex items-center justify-center text-xl font-bold">
            Click to Start
        </div>
        
        <div id="message" class="text-xl font-mono text-yellow-400 my-4 h-8"></div>
        <div class="text-2xl font-semibold">
            Best Time: 
            <span id="bestTimeDisplay" class="text-green-500">N/A</span>
        </div>
    </div>

    <script>
        const colorBox = document.getElementById('colorBox');
        const messageDisplay = document.getElementById('message');
        const bestTimeDisplay = document.getElementById('bestTimeDisplay');
        
        let state = 'ready'; // 'ready', 'waiting', 'go'
        let timeoutId = null;
        let startTime = 0;
        let bestTime = Infinity;

        function loadBestTime() {
            const storedTime = localStorage.getItem('clickerBestTime');
            if (storedTime) {
                bestTime = parseFloat(storedTime);
                bestTimeDisplay.textContent = `${bestTime.toFixed(2)} ms`;
            } else {
                bestTimeDisplay.textContent = 'N/A';
            }
        }

        function updateUI(text, color, boxText) {
            messageDisplay.textContent = text;
            colorBox.style.backgroundColor = color;
            colorBox.textContent = boxText || '';
        }

        function resetGame() {
            if (timeoutId) clearTimeout(timeoutId);
            state = 'ready';
            updateUI('Click to start waiting...', '#4a90e2', 'Click to Start');
            loadBestTime();
        }

        function handleBoxClick() {
            if (state === 'ready') {
                state = 'waiting';
                updateUI('Wait for GREEN...', '#e74c3c', 'Wait');
                
                const delay = Math.random() * 2500 + 1500; 
                
                timeoutId = setTimeout(() => {
                    state = 'go';
                    startTime = performance.now();
                    updateUI('CLICK NOW!', '#2ecc71', 'CLICK');
                }, delay);

            } else if (state === 'waiting') {
                resetGame();
                updateUI('Too early! Restarting in 2s...', '#f39c12', 'Restart');
                setTimeout(() => {
                    resetGame();
                }, 2000);

            } else if (state === 'go') {
                const reactionTime = performance.now() - startTime;
                
                let isNewBest = false;
                
                if (reactionTime < bestTime) {
                    bestTime = reactionTime;
                    localStorage.setItem('clickerBestTime', bestTime);
                    isNewBest = true;
                }
                
                updateUI(`Success! Time: ${reactionTime.toFixed(2)} ms. Click to try again.`, '#2ecc71', 'Play Again');
                
                bestTimeDisplay.textContent = `${bestTime.toFixed(2)} ms` + (isNewBest ? ' (NEW BEST!)' : '');

                state = 'ready'; 
                
            }
        }

        colorBox.addEventListener('click', handleBoxClick);
        resetGame();
    </script>
</body>
</html>
"""

# --- 4. Memory Match Game HTML/JS Content (Unchanged) ---

MEMORY_MATCH_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0d1117; }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            width: 480px;
            margin: 20px auto;
            padding: 18px;
            border-radius: 20px;
            background-color: #1e293b;
        }
        .card {
            width: 100px;
            height: 100px;
            background-color: #334155; /* Blue-gray background when hidden */
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            cursor: pointer;
            transition: transform 0.3s, background-color 0.3s;
            transform-style: preserve-3d;
            position: relative;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card.flipped {
            background-color: #e5e7eb; /* Light background when flipped */
            color: #1f2937;
            transform: rotateY(180deg);
        }
        .card.matched {
            background-color: #10b981; /* Green when matched */
            color: #ffffff;
            opacity: 0.8;
            pointer-events: none;
        }
        .card-inner {
            transform: rotateY(180deg);
            backface-visibility: hidden;
        }
        .card-front {
            position: absolute;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            backface-visibility: hidden;
        }
        .card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            transform: rotateY(180deg);
            backface-visibility: hidden;
        }
    </style>
</head>
<body class="p-4 md:p-8">

    <div class="max-w-xl mx-auto text-white text-center">
        <h2 class="text-3xl font-bold mb-2">🧠 Memory Match</h2>
        <p class="text-gray-400 mb-6">Match all pairs in the fewest moves possible!</p>
        
        <div class="grid-container" id="gameGrid">
            <!-- Cards go here -->
        </div>

        <div class="mt-6 w-full flex justify-between text-xl font-mono p-4 bg-gray-700 rounded-lg shadow-inner max-w-[420px] mx-auto">
            <div>Moves: <span id="movesDisplay" class="text-yellow-400 font-bold">0</span></div>
            <div>Best Score (Moves): <span id="bestScoreDisplay" class="text-green-400 font-bold">N/A</span></div>
        </div>
        
        <button id="startButton" class="mt-6 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-md">
            Start Game
        </button>
        
        <!-- Game Over Modal (Initially hidden) -->
        <div id="gameOverModal" class="fixed inset-0 bg-gray-900 bg-opacity-90 flex items-center justify-center hidden">
            <div class="bg-gray-800 p-10 rounded-xl shadow-2xl border-2 border-purple-500 text-center">
                <h3 class="text-4xl font-extrabold text-white mb-4">Congratulations!</h3>
                <div class="text-2xl text-gray-300 mb-6">You finished in <span id="finalMoves" class="font-extrabold text-yellow-400">0</span> moves.</div>
                <div id="newRecordMessage" class="text-xl text-green-400 font-bold mb-6 hidden">NEW RECORD!</div>
                <button onclick="startGame()" class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-md">
                    Play Again
                </button>
            </div>
        </div>
    </div>

    <script>
        const gameGrid = document.getElementById('gameGrid');
        const movesDisplay = document.getElementById('movesDisplay');
        const bestScoreDisplay = document.getElementById('bestScoreDisplay');
        const startButton = document.getElementById('startButton');
        const gameOverModal = document.getElementById('gameOverModal');
        const finalMovesDisplay = document.getElementById('finalMoves');
        const newRecordMessage = document.getElementById('newRecordMessage');

        const ICONS = ['⭐', '🚀', '💡', '🏆', '🧩', '💎', '🔑', '🌍'];
        const CARD_COUNT = ICONS.length * 2; // 16 cards (4x4 grid)

        let cards = [];
        let firstCard = null;
        let secondCard = null;
        let lockBoard = false;
        let moves = 0;
        let bestScore = Infinity;
        let matchedPairs = 0;

        function shuffle(array) {
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
            return array;
        }
        
        function loadBestScore() {
            const storedScore = localStorage.getItem('memoryBestScore');
            if (storedScore) {
                bestScore = parseInt(storedScore, 10);
                bestScoreDisplay.textContent = bestScore;
            } else {
                bestScoreDisplay.textContent = 'N/A';
                bestScore = Infinity;
            }
        }

        function updateMoves() {
            movesDisplay.textContent = moves;
        }

        function setupBoard() {
            const gameIcons = shuffle([...ICONS, ...ICONS]);

            gameGrid.innerHTML = '';
            cards = [];
            
            gameIcons.forEach((icon, index) => {
                const cardElement = document.createElement('div');
                cardElement.classList.add('card');
                cardElement.dataset.icon = icon;
                cardElement.dataset.index = index;
                
                cardElement.innerHTML = `
                    <div class="card-front">?</div>
                    <div class="card-back">${icon}</div>
                `;

                cardElement.addEventListener('click', () => handleCardClick(cardElement));
                gameGrid.appendChild(cardElement);
                cards.push(cardElement);
            });
        }
        
        function resetGame() {
            moves = 0;
            matchedPairs = 0;
            firstCard = null;
            secondCard = null;
            lockBoard = false;
            gameOverModal.classList.add('hidden');
            updateMoves();
            loadBestScore();
            setupBoard();
        }

        function handleCardClick(cardElement) {
            if (lockBoard) return;
            if (cardElement === firstCard) return;
            if (cardElement.classList.contains('flipped') || cardElement.classList.contains('matched')) return;

            cardElement.classList.add('flipped');

            if (!firstCard) {
                firstCard = cardElement;
                return;
            }

            secondCard = cardElement;
            moves++;
            updateMoves();
            lockBoard = true;

            checkForMatch();
        }

        function checkForMatch() {
            const isMatch = firstCard.dataset.icon === secondCard.dataset.icon;
            
            isMatch ? disableCards() : unflipCards();
        }

        function disableCards() {
            firstCard.classList.add('matched');
            secondCard.classList.add('matched');
            matchedPairs++;
            
            resetTurn();

            if (matchedPairs === ICONS.length) {
                endGame();
            }
        }

        function unflipCards() {
            setTimeout(() => {
                firstCard.classList.remove('flipped');
                secondCard.classList.remove('flipped');
                resetTurn();
            }, 1000); 
        }

        function resetTurn() {
            [firstCard, secondCard, lockBoard] = [null, null, false];
        }

        function endGame() {
            finalMovesDisplay.textContent = moves;
            
            let isNewRecord = false;

            // Lowest moves is the best score
            if (moves < bestScore) {
                bestScore = moves;
                localStorage.setItem('memoryBestScore', bestScore);
                bestScoreDisplay.textContent = bestScore;
                isNewRecord = true;
            }
            
            newRecordMessage.classList.toggle('hidden', !isNewRecord);
            
            gameOverModal.classList.remove('hidden');
        }

        window.startGame = () => {
             startButton.classList.add('hidden');
             resetGame();
        }
        
        loadBestScore();
        startButton.addEventListener('click', window.startGame);
        setupBoard();
    </script>
</body>
</html>
"""

# --- 5. Pacman Game HTML/JS Content ---

PACMAN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0d1117; }
        #pacmanCanvas {
            background-color: #000000;
            border: 4px solid #f39c12;
            box-shadow: 0 0 15px rgba(243, 156, 18, 0.7);
            display: block;
            margin: 0 auto;
            max-width: 95vw;
            border-radius: 12px;
        }
        .pacman-message-box {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 100;
            min-width: 280px;
        }
    </style>
</head>
<body class="p-4 md:p-8">

    <div class="max-w-4xl mx-auto">
        <h2 class="text-3xl font-bold text-white mb-4 text-center">🟡 Simple Pacman</h2>
        <p class="text-gray-400 text-center mb-6">
            Use **WASD** or **Arrow Keys** to eat all the dots! Avoid the blue ghost.
        </p>

        <div class="relative flex flex-col items-center">
            <canvas id="pacmanCanvas" width="400" height="400"></canvas>

            <!-- Message Box for Start/Game Over -->
            <div id="pacmanMessageBox" class="pacman-message-box bg-gray-800 p-8 rounded-xl shadow-2xl border-2 border-yellow-400 text-center transition duration-300">
                <h3 id="pacmanMessageText" class="text-3xl font-bold text-white mb-4">Game Over!</h3>
                <div class="text-xl text-gray-300 mb-6">Score: <span id="pacmanFinalScore" class="font-extrabold text-yellow-400">0</span></div>
                <button id="pacmanStartButton" class="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-md">
                    Start Game (Press Enter)
                </button>
            </div>

            <!-- Score HUD -->
            <div class="mt-4 w-full flex justify-between text-white text-xl font-mono p-4 bg-gray-700 rounded-lg shadow-inner max-w-[400px]">
                <div>Score: <span id="pacmanScoreDisplay" class="text-yellow-400 font-bold">0</span></div>
                <div>High Score: <span id="pacmanHighScoreDisplay" class="text-green-400 font-bold">0</span></div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('pacmanCanvas');
            const ctx = canvas.getContext('2d');
            const scoreDisplay = document.getElementById('pacmanScoreDisplay');
            const highScoreDisplay = document.getElementById('pacmanHighScoreDisplay');
            const messageBox = document.getElementById('pacmanMessageBox');
            const messageText = document.getElementById('pacmanMessageText');
            const finalScore = document.getElementById('pacmanFinalScore');
            const startButton = document.getElementById('pacmanStartButton');

            // Game Constants
            const GRID_SIZE = 20;
            const TILE_SIZE = canvas.width / GRID_SIZE;
            const DOT_RADIUS = TILE_SIZE / 8;
            const PLAYER_RADIUS = TILE_SIZE / 2;
            const GAME_SPEED = 150; // MODIFIED: Increased to 150ms (from 100ms) to slow the game down
            
            // Game State
            let pacman = { x: 1, y: 1, dx: 0, dy: 0, nextDx: 0, nextDy: 0, angle: 0 };
            let ghost = { x: GRID_SIZE - 2, y: GRID_SIZE - 2, color: '#3498db' };
            let dots = [];
            let score = 0;
            let highScore = 0;
            let gameLoopInterval;
            
            // 0: Wall, 1: Dot, 2: Empty/Path
            const maze = [
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,2,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0],
                [0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0],
                [0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0],
                [0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0],
                [0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0],
                [0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ];

            function loadHighScore() {
                const storedScore = localStorage.getItem('pacmanHighScore');
                highScore = storedScore ? parseInt(storedScore, 10) : 0;
            }

            function saveHighScore() {
                if (score > highScore) {
                    highScore = score;
                    localStorage.setItem('pacmanHighScore', highScore);
                }
            }
            
            function updateHUD() {
                scoreDisplay.textContent = score;
                highScoreDisplay.textContent = highScore;
            }

            function resetGame() {
                loadHighScore();
                pacman = { x: 1, y: 1, dx: 0, dy: 0, nextDx: 0, nextDy: 0, angle: 0 };
                ghost = { x: GRID_SIZE - 2, y: GRID_SIZE - 2, color: '#3498db' };
                score = 0;
                dots = [];

                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let x = 0; x < GRID_SIZE; x++) {
                        if (maze[y][x] === 1) {
                            dots.push({ x, y });
                        }
                    }
                }
                updateHUD();
            }

            function drawWall(x, y) {
                ctx.fillStyle = '#4a90e2';
                ctx.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            }

            function drawDot(x, y) {
                ctx.beginPath();
                ctx.fillStyle = '#f7dc6f';
                const centerX = x * TILE_SIZE + TILE_SIZE / 2;
                const centerY = y * TILE_SIZE + TILE_SIZE / 2;
                ctx.arc(centerX, centerY, DOT_RADIUS, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            function drawPacman() {
                const centerX = pacman.x * TILE_SIZE + TILE_SIZE / 2;
                const centerY = pacman.y * TILE_SIZE + TILE_SIZE / 2;
                
                // Determine mouth angle based on direction
                let startAngle = 0.25 * Math.PI + pacman.angle;
                let endAngle = 1.75 * Math.PI + pacman.angle;

                ctx.beginPath();
                ctx.fillStyle = '#f39c12';
                ctx.arc(centerX, centerY, PLAYER_RADIUS, startAngle, endAngle);
                ctx.lineTo(centerX, centerY);
                ctx.fill();
            }
            
            function drawGhost() {
                const centerX = ghost.x * TILE_SIZE + TILE_SIZE / 2;
                const centerY = ghost.y * TILE_SIZE + TILE_SIZE / 2;
                const width = TILE_SIZE * 0.8;
                const height = TILE_SIZE * 0.8;
                const halfWidth = width / 2;

                ctx.fillStyle = ghost.color;
                ctx.shadowColor = ghost.color;
                ctx.shadowBlur = 15;
                
                // Draw main body (rounded top, straight bottom)
                ctx.beginPath();
                ctx.arc(centerX, centerY - TILE_SIZE*0.1, halfWidth, Math.PI, 0, false);
                ctx.lineTo(centerX + halfWidth, centerY + height/2);
                ctx.lineTo(centerX - halfWidth, centerY + height/2);
                ctx.lineTo(centerX - halfWidth, centerY - TILE_SIZE*0.1);
                ctx.fill();
                
                ctx.shadowBlur = 0;
            }

            function isWall(x, y) {
                if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) {
                    return true;
                }
                return maze[y][x] === 0;
            }

            function movePacman() {
                let newX = pacman.x + pacman.nextDx;
                let newY = pacman.y + pacman.nextDy;
                
                // Check if the next intended move is valid
                if (!isWall(newX, newY)) {
                    pacman.dx = pacman.nextDx;
                    pacman.dy = pacman.nextDy;
                }
                
                // Move based on current direction
                newX = pacman.x + pacman.dx;
                newY = pacman.y + pacman.dy;

                if (!isWall(newX, newY)) {
                    pacman.x = newX;
                    pacman.y = newY;
                }
                
                // Update angle for drawing
                if (pacman.dx === 1) pacman.angle = 0;
                else if (pacman.dx === -1) pacman.angle = Math.PI;
                else if (pacman.dy === 1) pacman.angle = 0.5 * Math.PI;
                else if (pacman.dy === -1) pacman.angle = 1.5 * Math.PI;

                // Dot eating
                for (let i = 0; i < dots.length; i++) {
                    if (dots[i].x === pacman.x && dots[i].y === pacman.y) {
                        dots.splice(i, 1);
                        score += 10;
                        updateHUD();
                        break;
                    }
                }
                
                if (dots.length === 0) {
                    gameWin();
                }
            }

            function moveGhost() {
                const options = [];
                // Simple pursuit logic: try to move closer to Pacman
                
                if (!isWall(ghost.x + 1, ghost.y) && pacman.x > ghost.x) options.push({dx: 1, dy: 0});
                if (!isWall(ghost.x - 1, ghost.y) && pacman.x < ghost.x) options.push({dx: -1, dy: 0});
                if (!isWall(ghost.x, ghost.y + 1) && pacman.y > ghost.y) options.push({dx: 0, dy: 1});
                if (!isWall(ghost.x, ghost.y - 1) && pacman.y < ghost.y) options.push({dx: 0, dy: -1});
                
                // Add random moves if options are empty or for variation
                if (options.length === 0 || Math.random() < 0.3) {
                     // Add all valid non-wall moves
                    if (!isWall(ghost.x + 1, ghost.y)) options.push({dx: 1, dy: 0});
                    if (!isWall(ghost.x - 1, ghost.y)) options.push({dx: -1, dy: 0});
                    if (!isWall(ghost.x, ghost.y + 1)) options.push({dx: 0, dy: 1});
                    if (!isWall(ghost.x, ghost.y - 1)) options.push({dx: 0, dy: -1});
                }
                
                // Pick a move from available options
                if (options.length > 0) {
                    const move = options[Math.floor(Math.random() * options.length)];
                    ghost.x += move.dx;
                    ghost.y += move.dy;
                }
            }
            
            function checkCollision() {
                return pacman.x === ghost.x && pacman.y === ghost.y;
            }

            function gameLoop() {
                movePacman();
                
                // Ghost moves less frequently than Pacman (approx 2/3 speed)
                if (score % 20 === 0 && Math.random() < 0.7) { 
                    moveGhost();
                }

                if (checkCollision()) {
                    gameOver();
                    return;
                }

                // Drawing
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let x = 0; x < GRID_SIZE; x++) {
                        if (maze[y][x] === 0) {
                            drawWall(x, y);
                        }
                    }
                }
                
                dots.forEach(dot => drawDot(dot.x, dot.y));
                drawGhost();
                drawPacman();
                
                // Re-schedule the next frame
                gameLoopInterval = setTimeout(gameLoop, GAME_SPEED);
            }

            function handleInput(e) {
                if (messageBox.style.display !== 'none' && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    startGame();
                    return;
                }
                
                switch (e.key) {
                    case 'ArrowUp':
                    case 'w':
                        e.preventDefault();
                        pacman.nextDx = 0; pacman.nextDy = -1;
                        break;
                    case 'ArrowDown':
                    case 's':
                        e.preventDefault();
                        pacman.nextDx = 0; pacman.nextDy = 1;
                        break;
                    case 'ArrowLeft':
                    case 'a':
                        e.preventDefault();
                        pacman.nextDx = -1; pacman.nextDy = 0;
                        break;
                    case 'ArrowRight':
                    case 'd':
                        e.preventDefault();
                        pacman.nextDx = 1; pacman.nextDy = 0;
                        break;
                }
            }

            function gameOver(win = false) {
                clearTimeout(gameLoopInterval);
                saveHighScore();
                updateHUD();
                
                messageText.textContent = win ? "You Win!" : "Game Over!";
                finalScore.textContent = score;
                startButton.textContent = "Play Again";
                messageBox.style.display = 'block';
            }
            
            function gameWin() {
                // Award bonus for winning
                score += 500;
                gameOver(true);
            }

            function startGame() {
                messageBox.style.display = 'none';
                resetGame();
                gameLoop();
            }

            // --- Initial Setup ---
            document.addEventListener('keydown', handleInput);
            startButton.addEventListener('click', startGame);

            loadHighScore();
            gameOver(); 
            messageText.textContent = "Welcome to Pacman!";
            finalScore.textContent = 0;
        });
    </script>
</body>
</html>
"""

# --- Streamlit Python Logic ---

# Create the five tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🐍 Snake", "🚀 Obstacle Runner", "⚡ Color Clicker", "🧠 Memory Match", "🟡 Pacman"])

# --- Tab 1: Snake ---
with tab1:
    components.html(
        SNAKE_HTML,
        height=750,
    )

    st.markdown("""
    <div class="mt-8 text-gray-500 text-sm">
        <p>Maneuver the snake to consume food and grow longer, but be careful not to run into the walls or your own tail!</p>
    </div>
    """, unsafe_allow_html=True)


# --- Tab 2: Obstacle Runner ---
with tab2:
    components.html(
        OBSTACLE_RUNNER_HTML,
        height=650, 
    )

    st.markdown("""
    <div class="mt-8 text-gray-500 text-sm">
        <p>Navigate to avoid incoming obstacles that accelerate and spawn rapidly as your score increases! Test your reflexes and see how long you can last.</p>
    </div>
    """, unsafe_allow_html=True)


# --- Tab 3: Color Clicker ---
with tab3:
    components.html(
        COLOR_CLICKER_HTML,
        height=500,
    )
    st.markdown("""
    <div class="mt-4 text-gray-500 text-sm">
        <p>Test your reaction time by quickly clicking the target before the changing color confuses you. The lowest recorded time is your best score!</p>
    </div>
    """, unsafe_allow_html=True)

# --- Tab 4: Memory Match ---
with tab4:
    components.html(
        MEMORY_MATCH_HTML,
        height=750, 
    )
    st.markdown("""
    <div class="mt-4 text-gray-500 text-sm">
        <p>Flip pairs of matching tiles to clear the board in the fewest moves possible. Aim for the lowest score to prove your memory skills!</p>
    </div>
    """, unsafe_allow_html=True)

# --- Tab 5: Pacman ---
with tab5:
    components.html(
        PACMAN_HTML,
        height=650,
    )
    st.markdown("""
    <div class="mt-4 text-gray-500 text-sm">
        <p>Navigate the maze as Pacman, eat all the yellow dots, and avoid the blue ghost!</p>
    </div>
    """, unsafe_allow_html=True)