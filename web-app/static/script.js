let sections = document.querySelectorAll("section");
let header = document.querySelectorAll("header");

// REPLAY BUTTON
document.addEventListener("DOMContentLoaded", () => {
    const replayButton = document.getElementById("replayButton");

    replayButton.addEventListener("click", (event) => {
        event.preventDefault();
        console.log("Replay button clicked");
        resetGame(); // ADD RESET GAME LOGIC HERE
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
                cameraBox.innerHTML = ""; // Placeholder
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
                captureAndClassify(); // Replace generateRandomMove() with captureAndClassify()
            }
        }, 1000);
    }

    // Capture a frame and classify using backend API
    function captureAndClassify() {
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const context = canvas.getContext("2d");
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        const frameData = canvas.toDataURL("image/png");

        // Call backend /classify API
        fetch("/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({image_data: frameData}),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data && data.user_choice) {
                    const userChoice = data.user_choice;
                    generateComputerMove(userChoice); // Use classified result
                } else {
                    console.error("Invalid response from classify API:", data);
                    generateComputerMove(); // Fallback to random move
                }
            })
            .catch((err) => {
                console.error("Error classifying hand gesture:", err);
                generateComputerMove(); // Fallback to random move
            });
    }

    // Generate computer move and display results
    function generateComputerMove(userChoice = "Rock") {
        const moves = ["Rock", "Paper", "Scissors"];
        const computerChoice = moves[Math.floor(Math.random() * moves.length)];
        const winner = determineWinner(userChoice, computerChoice);

        // Display results
        displayResults(userChoice, computerChoice, winner);

        // Save game results to backend
        saveGameResults(userChoice, computerChoice, winner);
    }

    // Save game results to backend
    function saveGameResults(userChoice, computerChoice, winner) {
        fetch("/save_game_result", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_choice: userChoice,
                computer_choice: computerChoice,
                winner: winner,
                timestamp: new Date().toISOString(),
            }),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "success") {
                    console.log("Game result saved successfully:", data);
                } else {
                    console.error("Error saving game result:", data.message);
                }
            })
            .catch((err) => {
                console.error("Error saving game result:", err);
            });
    }

    // COMMENTED OUT: Original generateRandomMove function
    /*
    function generateRandomMove() {
        const moves = ["Rock", "Paper", "Scissors"];
        const userChoice = moves[Math.floor(Math.random() * moves.length)];
        const computerChoice = moves[Math.floor(Math.random() * moves.length)];
        const winner = determineWinner(userChoice, computerChoice);
        displayResults(userChoice, computerChoice, winner);
    }
    */

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