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

// === QUIZ DATA ===
const quizQuestions = [
  {
    question: "Wat is de kleur van een Smurf?",
    answers: { A: "Blauw", B: "Rood", C: "Groen", D: "Geel" },
    correct: "A"
  },
  {
    question: "Wat krijg je als je water kookt?",
    answers: { A: "IJs", B: "Stoom", C: "Melk", D: "Koffie" },
    correct: "B"
  },
  {
    question: "Welke vogel kan niet vliegen?",
    answers: { A: "Pinguïn", B: "Mus", C: "Merel", D: "Zwaluw" },
    correct: "A"
  },
  {
    question: "Wat is het tegenovergestelde van warm?",
    answers: { A: "Heet", B: "Koud", C: "Nat", D: "Droog" },
    correct: "B"
  },
  {
    question: "Hoeveel poten heeft een spin?",
    answers: { A: "6", B: "8", C: "10", D: "100" },
    correct: "B"
  },
  // ... (95 extra vragen worden hier toegevoegd, zie aparte instructie) ...
];

// === QUIZ STATE ===
let quizState = {
  questions: [], // 10 random vragen
  current: 0,
  score: 0,
  started: false,
  finished: false
};

// Voeg globale countdown state toe
let countdownActive = false;
let countdownValue = 5;
let countdownTimeout = null;
let pendingAnswer = null;

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
                            // Countdown logica
                            if (quizState.started && !quizState.finished && !countdownActive && ["A","B","C","D"].includes(label)) {
                                startCountdown(label);
                            }
                        }
                    });
                } catch (error) {
                    console.error('Prediction error:', error);
                    displayPose("Error in prediction", 0);
                }
            } else {
                if (!quizState.started && !quizState.finished) {
                    showStartQuizButton();
                } else {
                    displayPose("Laad eerst een model", 0);
                }
            }
        }
    } else {
        if (!quizState.started && !quizState.finished && modelLoaded) {
            showStartQuizButton();
        } else {
            displayPose("Geen hand gedetecteerd", 0);
        }
    }

    // Overlay countdown
    if (countdownActive) {
        canvasCtx.save();
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
        canvasCtx.font = 'bold 80px Arial';
        canvasCtx.fillStyle = 'rgba(0,0,0,0.7)';
        canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.fillStyle = '#fff';
        canvasCtx.textAlign = 'center';
        canvasCtx.fillText(countdownValue, canvasElement.width/2, canvasElement.height/2);
        canvasCtx.restore();
    }

    canvasCtx.restore();

    // Zorg dat de quizvraag altijd zichtbaar blijft tijdens de quiz
    if (quizState.started && !quizState.finished) {
        showCurrentQuestion();
    }

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

    if (!quizState.started && modelLoaded) {
        showStartQuizButton();
        return;
    }
    if (poseLabel === "Geen hand gedetecteerd") {
        display.textContent = poseLabel;
        return;
    }
    if (!modelLoaded) {
        display.textContent = "Laad eerst een model";
        return;
    }
    if (quizState.finished) {
        showQuizResult();
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
            document.getElementById('currentPoseDisplay').innerHTML = 
                `Model geladen met ${modelData.trainingsData.length} voorbeelden!<br><button onclick="startQuiz()" class="mdc-button mdc-button--raised">Start Quiz</button>`;
            
        } catch (error) {
            console.error('Error loading model:', error);
            document.getElementById('currentPoseDisplay').textContent = 
                'Fout bij laden model: ' + error.message;
        }
    };
    reader.readAsText(file);
}

function showStartQuizButton() {
    document.getElementById('currentPoseDisplay').innerHTML =
        `Model geladen!<br><button onclick="startQuiz()" class="mdc-button mdc-button--raised">Start Quiz</button>`;
}

// Countdown logica
function startCountdown(label) {
    countdownActive = true;
    countdownValue = 5;
    pendingAnswer = label;
    nextCountdownTick();
}

function nextCountdownTick() {
    if (countdownValue > 0) {
        countdownTimeout = setTimeout(() => {
            countdownValue--;
            nextCountdownTick();
        }, 1000);
    } else {
        countdownActive = false;
        handleQuizAnswer(pendingAnswer);
        pendingAnswer = null;
    }
}

// Pas startQuiz aan zodat direct de eerste vraag getoond wordt
function startQuiz() {
    quizState.questions = shuffleArray(quizQuestions).slice(0, 10);
    quizState.current = 0;
    quizState.score = 0;
    quizState.started = true;
    quizState.finished = false;
    showCurrentQuestion();
}

function showCurrentQuestion() {
  const display = document.getElementById('currentPoseDisplay');
  if (!quizState.started || quizState.finished) return;
  const q = quizState.questions[quizState.current];
  display.innerHTML = `<h3>Vraag ${quizState.current + 1} van 10</h3>
    <p>${q.question}</p>
    <ul style='list-style:none;padding:0;'>
      <li><b>A:</b> ${q.answers.A}</li>
      <li><b>B:</b> ${q.answers.B}</li>
      <li><b>C:</b> ${q.answers.C}</li>
      <li><b>D:</b> ${q.answers.D}</li>
    </ul>
    <p>Maak het juiste handgebaar voor jouw antwoord!</p>`;
}

function handleQuizAnswer(label) {
  if (!quizState.started || quizState.finished) return;
  const q = quizState.questions[quizState.current];
  if (["A","B","C","D"].includes(label)) {
    if (label === q.correct) quizState.score++;
    quizState.current++;
    if (quizState.current >= 10) {
      quizState.finished = true;
      showQuizResult();
    } else {
      showCurrentQuestion();
    }
  }
}

function showQuizResult() {
  const display = document.getElementById('currentPoseDisplay');
  display.innerHTML = `<h3>Quiz afgelopen!</h3>
    <p>Je had ${quizState.score} van de 10 vragen goed!</p>
    <button onclick="startQuiz()" class="mdc-button mdc-button--raised">Opnieuw spelen</button>`;
}

function shuffleArray(array) {
  return array.map(a => [Math.random(), a]).sort((a, b) => a[0] - b[0]).map(a => a[1]);
}

// (Voeg hier 95 extra vragen toe in hetzelfde format als de eerste 5)
quizQuestions.push(
  { question: "Wat is de hoofdstad van Nederland?", answers: { A: "Rotterdam", B: "Amsterdam", C: "Utrecht", D: "Den Haag" }, correct: "B" },
  { question: "Welke maand heeft 28 dagen?", answers: { A: "Februari", B: "Allemaal", C: "Januari", D: "December" }, correct: "B" },
  { question: "Wat eet een konijn graag?", answers: { A: "Wortel", B: "Banaan", C: "Brood", D: "Kaas" }, correct: "A" },
  { question: "Hoe noem je een babyhond?", answers: { A: "Kitten", B: "Kalfje", C: "Puppy", D: "Veulen" }, correct: "C" },
  { question: "Wat is het grootste land ter wereld?", answers: { A: "China", B: "VS", C: "Rusland", D: "Canada" }, correct: "C" },
  { question: "Welke kleur krijg je als je rood en geel mengt?", answers: { A: "Groen", B: "Oranje", C: "Paars", D: "Blauw" }, correct: "B" },
  { question: "Hoeveel dagen heeft een week?", answers: { A: "5", B: "6", C: "7", D: "8" }, correct: "C" },
  { question: "Wat is de naam van de sneeuwpop in Frozen?", answers: { A: "Olaf", B: "Elsa", C: "Anna", D: "Kristoff" }, correct: "A" },
  { question: "Wat is het tegenovergestelde van groot?", answers: { A: "Lang", B: "Klein", C: "Dik", D: "Dun" }, correct: "B" },
  { question: "Welke planeet staat het dichtst bij de zon?", answers: { A: "Venus", B: "Aarde", C: "Mercurius", D: "Mars" }, correct: "C" },
  { question: "Wat is de langste rivier van de wereld?", answers: { A: "Nijl", B: "Amazonerivier", C: "Yangtze", D: "Mississippi" }, correct: "B" },
  { question: "Hoeveel tanden heeft een volwassen mens normaal?", answers: { A: "28", B: "30", C: "32", D: "36" }, correct: "C" },
  { question: "Wat is de hoofdstad van Frankrijk?", answers: { A: "Lyon", B: "Parijs", C: "Marseille", D: "Nice" }, correct: "B" },
  { question: "Welke kleur heeft een banaan?", answers: { A: "Geel", B: "Rood", C: "Groen", D: "Blauw" }, correct: "A" },
  { question: "Wat is het grootste zoogdier?", answers: { A: "Olifant", B: "Blauwe vinvis", C: "Nijlpaard", D: "Walrus" }, correct: "B" },
  { question: "Hoeveel uur zitten er in een dag?", answers: { A: "12", B: "24", C: "36", D: "48" }, correct: "B" },
  { question: "Wat is de hoofdstad van België?", answers: { A: "Brussel", B: "Antwerpen", C: "Gent", D: "Luik" }, correct: "A" },
  { question: "Welke sport wordt gespeeld op Wimbledon?", answers: { A: "Voetbal", B: "Tennis", C: "Hockey", D: "Basketbal" }, correct: "B" },
  { question: "Wat is de grootste planeet in ons zonnestelsel?", answers: { A: "Aarde", B: "Mars", C: "Jupiter", D: "Saturnus" }, correct: "C" },
  { question: "Hoeveel kleuren heeft de regenboog?", answers: { A: "5", B: "6", C: "7", D: "8" }, correct: "C" },
  { question: "Wat is de hoofdstad van Duitsland?", answers: { A: "Berlijn", B: "München", C: "Hamburg", D: "Keulen" }, correct: "A" },
  { question: "Wat is het kleinste bot in het menselijk lichaam?", answers: { A: "Stijgbeugel", B: "Scheenbeen", C: "Spaakbeen", D: "Dijbeen" }, correct: "A" },
  { question: "Welke kaas is beroemd uit Nederland?", answers: { A: "Brie", B: "Gouda", C: "Cheddar", D: "Feta" }, correct: "B" },
  { question: "Wat is de hoofdstad van Italië?", answers: { A: "Rome", B: "Milaan", C: "Venetië", D: "Napels" }, correct: "A" },
  { question: "Hoeveel continenten zijn er?", answers: { A: "5", B: "6", C: "7", D: "8" }, correct: "C" },
  { question: "Wat is de grootste oceaan?", answers: { A: "Atlantische Oceaan", B: "Indische Oceaan", C: "Stille Oceaan", D: "Noordelijke IJszee" }, correct: "C" },
  { question: "Welke planeet wordt de rode planeet genoemd?", answers: { A: "Mars", B: "Jupiter", C: "Venus", D: "Saturnus" }, correct: "A" },
  { question: "Wat is de hoofdstad van Spanje?", answers: { A: "Barcelona", B: "Madrid", C: "Valencia", D: "Sevilla" }, correct: "B" },
  { question: "Wat is het grootste eiland ter wereld?", answers: { A: "Groenland", B: "Australië", C: "Madagaskar", D: "IJsland" }, correct: "A" },
  { question: "Hoeveel seconden zitten er in een minuut?", answers: { A: "60", B: "100", C: "30", D: "90" }, correct: "A" },
  { question: "Wat is de hoofdstad van Engeland?", answers: { A: "Londen", B: "Manchester", C: "Liverpool", D: "Birmingham" }, correct: "A" },
  { question: "Welke kleur krijg je als je blauw en geel mengt?", answers: { A: "Groen", B: "Paars", C: "Oranje", D: "Bruin" }, correct: "A" },
  { question: "Wat is de grootste woestijn ter wereld?", answers: { A: "Sahara", B: "Gobi", C: "Kalahari", D: "Arabische Woestijn" }, correct: "A" },
  { question: "Hoeveel meter is een kilometer?", answers: { A: "100", B: "1000", C: "10.000", D: "1.000.000" }, correct: "B" },
  { question: "Wat is de hoofdstad van Portugal?", answers: { A: "Lissabon", B: "Porto", C: "Faro", D: "Braga" }, correct: "A" },
  { question: "Welke planeet heeft ringen?", answers: { A: "Mars", B: "Venus", C: "Saturnus", D: "Mercurius" }, correct: "C" },
  { question: "Wat is de grootste stad van Nederland?", answers: { A: "Rotterdam", B: "Amsterdam", C: "Den Haag", D: "Utrecht" }, correct: "B" },
  { question: "Hoeveel maanden heeft een jaar?", answers: { A: "10", B: "11", C: "12", D: "13" }, correct: "C" },
  { question: "Wat is de hoofdstad van Zweden?", answers: { A: "Stockholm", B: "Göteborg", C: "Malmö", D: "Uppsala" }, correct: "A" },
  { question: "Welke kleur heeft gras meestal?", answers: { A: "Blauw", B: "Groen", C: "Geel", D: "Rood" }, correct: "B" },
  { question: "Wat is de hoofdstad van Noorwegen?", answers: { A: "Oslo", B: "Bergen", C: "Trondheim", D: "Stavanger" }, correct: "A" },
  { question: "Hoeveel minuten zitten er in een uur?", answers: { A: "30", B: "45", C: "60", D: "90" }, correct: "C" },
  { question: "Wat is de hoofdstad van Denemarken?", answers: { A: "Kopenhagen", B: "Aarhus", C: "Odense", D: "Aalborg" }, correct: "A" },
  { question: "Welke kleur heeft een citroen?", answers: { A: "Geel", B: "Groen", C: "Rood", D: "Blauw" }, correct: "A" },
  { question: "Wat is de hoofdstad van Finland?", answers: { A: "Helsinki", B: "Espoo", C: "Tampere", D: "Turku" }, correct: "A" },
  { question: "Hoeveel dagen heeft februari in een schrikkeljaar?", answers: { A: "28", B: "29", C: "30", D: "31" }, correct: "B" },
  { question: "Wat is de hoofdstad van Oostenrijk?", answers: { A: "Wenen", B: "Salzburg", C: "Innsbruck", D: "Graz" }, correct: "A" },
  { question: "Welke kleur heeft een sinaasappel?", answers: { A: "Oranje", B: "Geel", C: "Groen", D: "Rood" }, correct: "A" },
  { question: "Wat is de hoofdstad van Zwitserland?", answers: { A: "Bern", B: "Zürich", C: "Genève", D: "Basel" }, correct: "A" },
  { question: "Hoeveel poten heeft een mier?", answers: { A: "4", B: "6", C: "8", D: "10" }, correct: "B" },
  { question: "Wat is de hoofdstad van Griekenland?", answers: { A: "Athene", B: "Thessaloniki", C: "Patras", D: "Heraklion" }, correct: "A" },
  { question: "Welke kleur heeft een tomaat?", answers: { A: "Rood", B: "Geel", C: "Groen", D: "Blauw" }, correct: "A" },
  { question: "Wat is de hoofdstad van Turkije?", answers: { A: "Istanbul", B: "Ankara", C: "Izmir", D: "Antalya" }, correct: "B" },
  { question: "Hoeveel tanden heeft een haai?", answers: { A: "10", B: "50", C: "300", D: "1000" }, correct: "C" },
  { question: "Wat is de hoofdstad van Rusland?", answers: { A: "Moskou", B: "Sint-Petersburg", C: "Novosibirsk", D: "Jekaterinenburg" }, correct: "A" },
  { question: "Welke kleur heeft een appel meestal?", answers: { A: "Rood", B: "Groen", C: "Geel", D: "Allemaal" }, correct: "D" },
  { question: "Wat is de hoofdstad van Polen?", answers: { A: "Warschau", B: "Krakau", C: "Gdansk", D: "Poznan" }, correct: "A" },
  { question: "Hoeveel ogen heeft een dobbelsteen?", answers: { A: "4", B: "5", C: "6", D: "7" }, correct: "C" },
  { question: "Wat is de hoofdstad van Tsjechië?", answers: { A: "Praag", B: "Brno", C: "Ostrava", D: "Plzen" }, correct: "A" },
  { question: "Welke kleur heeft een wortel?", answers: { A: "Oranje", B: "Geel", C: "Groen", D: "Rood" }, correct: "A" },
  { question: "Wat is de hoofdstad van Hongarije?", answers: { A: "Boedapest", B: "Debrecen", C: "Szeged", D: "Pécs" }, correct: "A" },
  { question: "Hoeveel vingers heeft een mens aan één hand?", answers: { A: "4", B: "5", C: "6", D: "7" }, correct: "B" },
  { question: "Wat is de hoofdstad van Ierland?", answers: { A: "Dublin", B: "Cork", C: "Galway", D: "Limerick" }, correct: "A" },
  { question: "Welke kleur heeft een aubergine?", answers: { A: "Paars", B: "Groen", C: "Geel", D: "Rood" }, correct: "A" },
  { question: "Wat is de hoofdstad van Schotland?", answers: { A: "Edinburgh", B: "Glasgow", C: "Aberdeen", D: "Dundee" }, correct: "A" },
  { question: "Hoeveel benen heeft een octopus?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "B" },
  { question: "Wat is de hoofdstad van Luxemburg?", answers: { A: "Luxemburg", B: "Esch", C: "Diekirch", D: "Grevenmacher" }, correct: "A" },
  { question: "Welke kleur heeft een flamingo?", answers: { A: "Roze", B: "Blauw", C: "Groen", D: "Geel" }, correct: "A" },
  { question: "Wat is de hoofdstad van Slovenië?", answers: { A: "Ljubljana", B: "Maribor", C: "Celje", D: "Koper" }, correct: "A" },
  { question: "Hoeveel tanden heeft een volwassen kat?", answers: { A: "20", B: "26", C: "30", D: "32" }, correct: "C" },
  { question: "Wat is de hoofdstad van Slowakije?", answers: { A: "Bratislava", B: "Kosice", C: "Presov", D: "Nitra" }, correct: "A" },
  { question: "Welke kleur heeft een citroen?", answers: { A: "Geel", B: "Groen", C: "Rood", D: "Blauw" }, correct: "A" },
  { question: "Wat is de hoofdstad van Kroatië?", answers: { A: "Zagreb", B: "Split", C: "Rijeka", D: "Osijek" }, correct: "A" },
  { question: "Hoeveel poten heeft een krab?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "C" },
  { question: "Wat is de hoofdstad van Servië?", answers: { A: "Belgrado", B: "Novi Sad", C: "Nis", D: "Kragujevac" }, correct: "A" },
  { question: "Welke kleur heeft een olifant meestal?", answers: { A: "Grijs", B: "Bruin", C: "Zwart", D: "Wit" }, correct: "A" },
  { question: "Wat is de hoofdstad van Roemenië?", answers: { A: "Boekarest", B: "Cluj", C: "Timisoara", D: "Iasi" }, correct: "A" },
  { question: "Hoeveel poten heeft een spin?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "B" },
  { question: "Wat is de hoofdstad van Bulgarije?", answers: { A: "Sofia", B: "Plovdiv", C: "Varna", D: "Burgas" }, correct: "A" },
  { question: "Welke kleur heeft een panda?", answers: { A: "Zwart-wit", B: "Bruin", C: "Grijs", D: "Rood" }, correct: "A" },
  { question: "Wat is de hoofdstad van Estland?", answers: { A: "Tallinn", B: "Tartu", C: "Narva", D: "Parnu" }, correct: "A" },
  { question: "Hoeveel poten heeft een spin?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "B" },
  { question: "Wat is de hoofdstad van Letland?", answers: { A: "Riga", B: "Daugavpils", C: "Liepaja", D: "Jelgava" }, correct: "A" },
  { question: "Welke kleur heeft een zonnebloem?", answers: { A: "Geel", B: "Rood", C: "Blauw", D: "Groen" }, correct: "A" },
  { question: "Wat is de hoofdstad van Litouwen?", answers: { A: "Vilnius", B: "Kaunas", C: "Klaipeda", D: "Siauliai" }, correct: "A" },
  { question: "Hoeveel poten heeft een spin?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "B" },
  { question: "Wat is de hoofdstad van IJsland?", answers: { A: "Reykjavik", B: "Akureyri", C: "Kopavogur", D: "Hafnarfjordur" }, correct: "A" },
  { question: "Welke kleur heeft een walvis?", answers: { A: "Blauw", B: "Groen", C: "Rood", D: "Geel" }, correct: "A" },
  { question: "Wat is de hoofdstad van Malta?", answers: { A: "Valletta", B: "Birkirkara", C: "Mosta", D: "Qormi" }, correct: "A" },
  { question: "Hoeveel poten heeft een spin?", answers: { A: "6", B: "8", C: "10", D: "12" }, correct: "B" }
);

window.startQuiz = startQuiz;
window.showStartQuizButton = showStartQuizButton;
