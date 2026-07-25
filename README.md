# SafeAI-Agents

A modular **multi-agent AI system** for safe robotic manipulation and control. The project demonstrates how specialized AI agents can collaborate to interpret a scene, generate action plans, control robot movement, and enforce safety constraints before execution.

The system consists of four cooperating agents:

- 👁️ **Vision Agent** – Understands the environment.
- 🧠 **Planning Agent** – Converts natural language tasks into executable plans.
- 🎮 **Control Agent** – Generates robot control actions using a neural network.
- 🛡️ **Safety Monitor** – Independently verifies every action before execution.

An interactive **Streamlit dashboard** visualizes the complete decision-making pipeline, allowing users to inspect how each agent contributes throughout the simulation.

---

## Overview

The project was originally developed as a research notebook exploring:

- Multi-agent AI systems
- Vision-language grounding
- Safe robotic control
- AI safety verification

This repository reorganizes that work into a clean, deployable web application with an interactive interface suitable for demonstrations, experimentation, and future research.

---

# Features

- Multi-agent robotic control pipeline
- Natural language task execution
- Two-stage safety verification
- Neural-network-based controller
- Interactive 3D trajectory visualization
- Agent reasoning console
- Modular architecture for future model replacement
- Streamlit dashboard

---

# System Workflow

Given a natural language instruction such as:

> **"Pick and place the object while avoiding obstacles."**

the system performs the following sequence:

1. Understands the scene and identifies relevant objects.
2. Converts the instruction into an ordered action plan.
3. Validates the plan against predefined safety rules.
4. Generates robot control actions at every simulation step.
5. Performs a second independent safety verification before execution.
6. Updates the simulated environment.
7. Records the trajectory, rewards, safety decisions, and agent logs.

The output includes:

- Robot trajectory
- Safety decisions
- Agent communication logs
- Performance metrics
- Interactive visualizations

---

# Architecture

The system follows a modular multi-agent architecture.

```
                Natural Language Task
                        │
                        ▼
               ┌──────────────────┐
               │   Vision Agent   │
               └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Planning Agent   │
               │ (Safety Check #1)│
               └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Control Agent    │
               └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Safety Monitor   │
               │ (Safety Check #2)│
               └──────────────────┘
                        │
             Approved / Rejected Action
                        │
                        ▼
               Simulation Environment
```

---

# Agents

## 👁️ Vision Agent

Responsible for scene understanding.

The vision agent analyzes the current environment and estimates which objects are present by comparing candidate descriptions such as:

- red block
- obstacle
- empty space
- target object

### Current implementation

The original research notebook used **CLIP** for vision-language grounding.

To keep deployment lightweight, this implementation replaces CLIP with a deterministic seeded scoring function that mimics the same interface while avoiding large model downloads.

Because the interface remains unchanged, a real vision-language model can later replace this implementation without modifying the rest of the pipeline.

---

## 🧠 Planning Agent

Responsible for task planning.

The planning agent converts natural language instructions into an ordered sequence of robot actions.

Example plan:

```
Locate object
Move toward object
Grasp object
Move toward target
Release object
```

It selects planning templates based on task keywords such as:

- Pick and place
- Obstacle avoidance
- Exploration

### Safety Validation

Before execution, every action is checked against predefined constraints including:

- Maximum force
- Maximum speed
- Valid target positions

Unsafe actions are removed before reaching the controller.

This represents the **first safety layer** in the architecture.

---

## 🎮 Control Agent

Responsible for robot motion control.

The controller is implemented as a lightweight **PyTorch feed-forward neural network**.

Input:

- Position
- Orientation

(6-dimensional robot state)

Output:

- Movement direction
- Movement magnitude

(3-dimensional bounded action)

The network uses a **tanh** activation to ensure all outputs remain within the interval:

```
[-1, 1]
```

Unlike reinforcement learning controllers, this implementation produces stable and deterministic control actions suitable for demonstration purposes.

---

## 🛡️ Safety Monitor

Responsible for execution-time safety.

Before every control action is applied, the safety monitor evaluates the combined:

- robot state
- proposed action

using an independent neural classifier.

If the predicted safety score falls below a predefined threshold:

- the action is rejected
- the robot remains stationary

This represents the **second independent safety layer**, ensuring that even a valid high-level plan can be stopped if execution becomes unsafe.

---

# Simulation Loop

Each simulation step performs the following operations:

```
Observe Environment
        │
        ▼
Generate Control Action
        │
        ▼
Safety Verification
        │
        ▼
 Approved?
   │      │
 Yes      No
 │         │
 ▼         ▼
Move    Reject Action
 │
 ▼
Update Environment
```

The simulation continues until:

- the goal is reached, or
- the maximum number of simulation steps is exceeded.

The system records:

- robot trajectory
- cumulative reward
- safety rejections
- collisions
- distance to target

---

# Streamlit Dashboard

The application provides three interactive pages.

## Architecture

Displays:

- System overview
- Agent responsibilities
- Data flow between agents

---

## Simulation

Displays:

- Total reward
- Number of steps
- Safety rejections
- Collision count
- Interactive 3D robot trajectory
- Distance-to-target graph
- Vision confidence scores
- Approved action plan

---

## Agent Console

Displays a live log of every message generated by each agent.

Logs are color-coded by severity:

- Information
- Warning
- Safety
- Success

This makes every decision made during execution fully transparent.

---

# Configuration

The sidebar allows users to configure:

- Natural language task
- Maximum simulation steps
- Random seed

This enables reproducible experiments without modifying the source code.

---

# Project Structure

```
safeai_agents/
│
├── app.py              # Streamlit application
├── requirements.txt    # Project dependencies
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/safeai-agents.git

cd safeai-agents
```

---

## 2. Create a virtual environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Deployment

The application can be deployed using **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select the repository.
5. Set the entry point to:

```
app.py
```

6. Deploy.

---

# Design Decisions

The original research prototype relied on:

- CLIP for vision-language understanding
- TensorFlow for the execution-time safety classifier

While accurate, these dependencies significantly increased deployment time and resource requirements.

To improve portability, this implementation:

- uses **PyTorch** as the single deep-learning backend,
- replaces CLIP with a lightweight vision approximation,
- preserves identical interfaces between agents.

This modular design allows more sophisticated models to be integrated later without changing the surrounding architecture.

---

# Future Improvements

Potential extensions include:

- Real CLIP integration
- Large Language Model planning agent
- Reinforcement learning controller
- Physics-based robot simulation
- ROS integration
- Real robot deployment
- Dynamic obstacle environments
- Multi-robot coordination

---

# Technologies

- Python
- PyTorch
- Streamlit
- NumPy
- Plotly

---

# Research Focus

This project explores concepts at the intersection of:

- Multi-Agent AI
- Safe AI
- AI Alignment
- Robotics
- Neural Control
- Vision-Language Models
- Autonomous Systems

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This project builds upon ideas from research in:

- Agentic AI
- Vision-language models
- Safe robotic control
- AI safety and alignment

and was originally prototyped as an experimental research notebook before being refactored into this modular web application.