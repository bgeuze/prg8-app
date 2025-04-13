import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = undefined;
let webcamRunning = false;
let video = null;
let canvasElement = null;
let canvasCtx = null;
let lastVideoTime = -1;
let results = undefined;
let modelLoaded = false;
let knnClassifier = ml5.KNNClassifier();

let gameState = {
    round: 0,
    maxRounds: 3,
    score: {
        player: 0,
        computer: 0
    },
    playerMadeMove: false,
    gameOver: false
};

// Initialize everything after DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Initialize handLandmarker first
        await createHandLandmarker();
        console.log("HandLandmarker initialized successfully");

        // Initialize webcam button
        const webcamButton = document.getElementById("webcamButton");
        if (webcamButton) {
            webcamButton.addEventListener("click", enableCam);
        }

        // Initialize load model button
        const loadModelButton = document.getElementById('loadModelButton');
        if (loadModelButton) {
            loadModelButton.addEventListener('click', () => {
                fileInput.click();
            });
        }

        // Initialize video and canvas elements
        video = document.getElementById("webcam");
        canvasElement = document.getElementById("output_canvas");
        canvasCtx = canvasElement.getContext("2d");

        document.getElementById('currentPoseDisplay').textContent = "Systeem geïnitialiseerd. Laad een model om te beginnen.";
    } catch (error) {
        console.error("Error during initialization:", error);
        document.getElementById('currentPoseDisplay').textContent = "Error tijdens initialisatie: " + error.message;
    }
});

// Create the HandLandmarker
async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
}

// Enable the live webcam view and start detection
async function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! handLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        document.getElementById('webcamButton').innerText = "ENABLE WEBCAM";
        // Stop all tracks
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
    } else {
        webcamRunning = true;
        document.getElementById('webcamButton').innerText = "DISABLE WEBCAM";
    }

    // getUsermedia parameters
    const constraints = {
        video: {
            width: 640,
            height: 480
        }
    };

    // Activate the webcam stream
    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadedmetadata", () => {
            video.play();
            predictWebcam();
        });
    } catch (err) {
        console.error("Error accessing webcam:", err);
        document.getElementById('currentPoseDisplay').textContent = "Error accessing webcam: " + err.message;
    }
}

// Predict from webcam
async function predictWebcam() {
    if (!handLandmarker || !webcamRunning) {
        return;
    }

    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#FFFFFF",
                lineWidth: 5
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: "#000000",
                lineWidth: 2
            });

            if (modelLoaded) {
                try {
                    const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
                    knnClassifier.classify(pose, (error, result) => {
                        if (error) {
                            console.error('Classification error:', error);
                            displayPose("Error: " + error.message, 0);
                        } else {
                            const { label, confidencesByLabel } = result;
                            const confidence = confidencesByLabel[label] * 100;
                            displayPose(label, confidence);
                        }
                    });
                } catch (error) {
                    console.error('Prediction error:', error);
                    displayPose("Error in prediction", 0);
                }
            } else {
                displayPose("Laad eerst een model", 0);
            }
        }
    } else {
        displayPose("Geen hand gedetecteerd", 0);
    }

    canvasCtx.restore();

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Get computer choice
function getComputerChoice() {
    const choices = ['rock', 'paper', 'scissors'];
    return choices[Math.floor(Math.random() * choices.length)];
}

// Determine winner
function determineWinner(playerChoice, computerChoice) {
    if (playerChoice === computerChoice) return 'tie';
    
    if (
        (playerChoice === 'rock' && computerChoice === 'scissors') ||
        (playerChoice === 'paper' && computerChoice === 'rock') ||
        (playerChoice === 'scissors' && computerChoice === 'paper')
    ) {
        return 'player';
    }
    
    return 'computer';
}

// Get final winner
function getFinalWinner() {
    if (gameState.score.player > gameState.score.computer) {
        return 'Gefeliciteerd! Jij wint het spel! 🎉';
    } else if (gameState.score.player < gameState.score.computer) {
        return 'Game Over - Computer wint het spel! 🤖';
    }
    return 'Gelijkspel! 🤝';
}

// Reset game
function resetGame() {
    gameState = {
        round: 0,
        maxRounds: 3,
        score: {
            player: 0,
            computer: 0
        },
        playerMadeMove: false,
        gameOver: false
    };
}

// Display pose and game results
function displayPose(poseLabel, confidence) {
    const display = document.getElementById('currentPoseDisplay');
    if (!display) return;

    if (poseLabel === "Geen hand gedetecteerd") {
        display.textContent = poseLabel;
        return;
    }

    if (!modelLoaded) {
        display.textContent = "Laad eerst een model";
        return;
    }

    if (gameState.gameOver) {
        display.innerHTML = `
            <h3>Spel Afgelopen!</h3>
            <p>${getFinalWinner()}</p>
            <p>Eindstand - Jij: ${gameState.score.player} Computer: ${gameState.score.computer}</p>
            <button onclick="resetGame()" class="mdc-button mdc-button--raised">
                Nieuw Spel
            </button>
        `;
        return;
    }

    if (!gameState.playerMadeMove) {
        // Player's turn
        gameState.playerMadeMove = true;
        const computerChoice = getComputerChoice();
        const winner = determineWinner(poseLabel, computerChoice);

        // Update score
        if (winner === 'player') gameState.score.player++;
        else if (winner === 'computer') gameState.score.computer++;

        // Update round
        gameState.round++;

        // Display round result
        let result = `<h3>Ronde ${gameState.round} van ${gameState.maxRounds}</h3>`;
        result += `Jouw keuze: ${poseLabel} (${confidence.toFixed(2)}%)<br>`;
        result += `Computer keuze: ${computerChoice}<br>`;
        result += `Resultaat: ${winner === 'tie' ? 'Gelijkspel!' : winner === 'player' ? 'Jij wint!' : 'Computer wint!'}<br>`;
        result += `Score - Jij: ${gameState.score.player} Computer: ${gameState.score.computer}`;

        if (gameState.round >= gameState.maxRounds) {
            gameState.gameOver = true;
            result += `<br><br><strong>${getFinalWinner()}</strong>`;
            result += `<br><button onclick="resetGame()" class="mdc-button mdc-button--raised">
                Nieuw Spel
            </button>`;
        } else {
            result += '<br><br>Maak een nieuw gebaar voor de volgende ronde!';
            // Reset for next round
            setTimeout(() => {
                gameState.playerMadeMove = false;
            }, 2000);
        }

        display.innerHTML = result;
    }
}

// Add file input for model loading
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.json';
fileInput.style.display = 'none';
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        loadModel(e.target.files[0]);
    }
});
document.body.appendChild(fileInput);

// Function to load the model
function loadModel(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const modelData = JSON.parse(e.target.result);
            console.log('Loading model data:', modelData);
            
            // Clear existing classifier
            knnClassifier.clearAllLabels();
            
            // Add all examples to the classifier
            modelData.trainingsData.forEach(({ pose, label }) => {
                knnClassifier.addExample(pose, label);
                console.log(`Added example for ${label}`);
            });
            
            modelLoaded = true;
            document.getElementById('currentPoseDisplay').textContent = 
                `Model geladen met ${modelData.trainingsData.length} voorbeelden!`;
            
        } catch (error) {
            console.error('Error loading model:', error);
            document.getElementById('currentPoseDisplay').textContent = 
                'Fout bij laden model: ' + error.message;
        }
    };
    reader.readAsText(file);
}
