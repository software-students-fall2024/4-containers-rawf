![Machine Learning Client](https://github.com/software-students-fall2024/4-containers-rawf/actions/workflows/ml-client-ci.yml/badge.svg)

![Web App](https://github.com/software-students-fall2024/4-containers-rawf/actions/workflows/web-app-ci.yml/badge.svg)

![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# CONTAINERIZED APP EXERCISE

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## TABLE OF CONTENTS
1. [Description](#description)
3. [Setup Steps](#setup-steps)
3. [Team Members](#team-members)
4. [Task Board](https://github.com/orgs/software-students-fall2024/projects/120)

## DESCRIPTION

Our containerized web app is an interactive game of *Rock-Paper-Scissors* between the user and computer by utilizing machine learning for image recognition. Our web app leverages Docker to containerize the application, making usage and management simple and easy across different environments.

***Key Features:***

**1. Image Recognition with Machine Learning:**

- Uses a pre-trained machine learning model to recognize hand gestures (rock, paper, scissors) via the user's webcam
- Processes images in real-time to identify the user's choice 
- Compares predicted user choice with computer's randomly generated choice to determine the winner of current round
 
**2. Web-Based User Interface:**

- Provides an intuitive and responsive web interface that guides users through the game
- Easily play the game directly from your web browser without needing to install any additional software
- Provides real-time feedback, with visual cues showing the user's move, computer's move, and outcome of each round

## SETUP STEPS

**1. Clone the Repository**
```
git clone <[repository-url](https://github.com/software-students-fall2024/4-containers-rawf)>
cd <4-containers-rawf>
```

**2. Verify MongoDB Connection:**
- *Download MongoDB:* download this extension onto your chosen source code editor
- *Connect to Database URL:* mongodb://mongodb:27017/

**3. Create a Virtual Environment:**
- *Mac:*
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

- *Windows:*
   ```
   python3 -m venv .venv
   .venv\Scripts\activate
   ```

**4. Install Dependencies:**
```
pip install requests
pip install pymongo
```

**5. Integrate with Docker Compose:** 
```
docker-compose up --build
```

**6. Play:** access our web app, [Rock, Paper, Scissors](http://localhost:5000), and play! 

## TEAM MEMBERS

- [Wenli Shi](https://github.com/WenliShi2332)
- [Alex Hsu](https://github.com/hsualexotake)
- [Reese Burns](https://github.com/reeseburns)
- [Finn Eskeland](https://github.com/finn1003)
