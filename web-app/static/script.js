let sections = document.querySelectorAll("section");
let header = document.querySelectorAll("header");

// REPLAY BUTTON
document.addEventListener("DOMContentLoaded", () => {
    const replayButton = document.getElementById("replayButton");

    replayButton.addEventListener("click", (event) => {
        event.preventDefault();
        // ADD RESET GAME LOGIC HERE
        console.log("Replay button clicked");
        resetGame();
    });
});

// GAME PLAYTHROUGH (CAMERA BOX, COUNTDOWN, CURRENT STATS)
document.addEventListener("DOMContentLoaded", () => {
    // Define variables
    const cameraBox = document.getElementById("camera-box");
    const countdownDisplay = document.getElementById("countdown");
    const yesButton = document.getElementById("yes-button");
    const videoElement = document.createElement("video");
    let cameraInitialized = false;

    const choicesDisplay = document.createElement("div");
    choicesDisplay.setAttribute("id", "choices");
    choicesDisplay.style.display = "none";
    cameraBox.after(choicesDisplay);

    videoElement.setAttribute("autoplay", true);
    videoElement.setAttribute("playsinline", true);
    videoElement.setAttribute("muted", true);

    // Start camera
    function startCamera() {
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then((stream) => {
                videoElement.srcObject = stream;
                cameraBox.innerHTML = "";            // Placeholder
                cameraBox.appendChild(videoElement);
                cameraInitialized = true;
                startGameCountdown();
            })
            .catch((err) => {
                console.error("Error accessing the camera:", err);
                cameraBox.innerHTML = "Unable to access the camera. Playing with random moves.";
                cameraInitialized = false;
                startGameCountdown();
            });
    }

    // Start game timer
    function startGameCountdown() {
        const countdownTexts = ["Rock...", "Paper...", "Scissors...", "Shoot!"];
        let index = 0;

        countdownDisplay.style.display = "block";
        choicesDisplay.style.display = "none";

        const interval = setInterval(() => {
            countdownDisplay.innerText = countdownTexts[index];
            index++;

            if (index === countdownTexts.length) {
                clearInterval(interval);
                generateRandomMove();
            }
        }, 1000);
    }

    // Generate random move and display results
    function generateRandomMove() {
        const moves = ["Rock", "Paper", "Scissors"];
        const userChoice = moves[Math.floor(Math.random() * moves.length)];
        const computerChoice = moves[Math.floor(Math.random() * moves.length)];
        const winner = determineWinner(userChoice, computerChoice);
        displayResults(userChoice, computerChoice, winner);
    }

    // Determine winner
    function determineWinner(userChoice, computerChoice) {
        if (userChoice === computerChoice) {return "Nobody";} // Tie

        // User wins
        else if (
            (userChoice === "Rock" && computerChoice === "Scissors") ||
            (userChoice === "Paper" && computerChoice === "Rock") ||
            (userChoice === "Scissors" && computerChoice === "Paper"))
            {return "User";}

        else {return "Computer";} // Computer wins
    }

    // Display results
    function displayResults(userChoice, computerChoice, winner) {
        countdownDisplay.style.display = "none"; // Hide countdown
        choicesDisplay.style.display = "block";  // Show results

        // Determine >, <, =
        let comparisonLine = "";
        if (winner === "User") {
            comparisonLine = `<strong>${userChoice}</strong> > ${computerChoice}`;
        } else if (winner === "Computer") {
            comparisonLine = `${userChoice} < <strong>${computerChoice}</strong>`;
        } else {
            comparisonLine = `${userChoice} <strong>=</strong> ${computerChoice}`;
        }

        // Print results
        choicesDisplay.innerHTML = `
            <p>You chose <strong>${userChoice}</strong> & Computer chose <strong>${computerChoice}</strong>.</p>
            <p>${comparisonLine}</p>
            <p><strong>${winner}</strong> wins!</p>
        `;
    }

    // IFF game played once...
    yesButton.addEventListener("click", () => {
        startCamera();
        yesButton.textContent = "Play again!";
    });
});