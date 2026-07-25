"""
PhysicalAI-Agents — Streamlit Dashboard
Multi-Agent System for Robotic Manipulation & Control
Converted from the original Colab notebook into an interactive app.
"""

import time
from collections import deque
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================

st.set_page_config(
    page_title="SafeAI-Agents",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg-deep:    #0B0F14;
    --bg-panel:   #121821;
    --bg-panel-2: #171F2B;
    --line:       #223047;
    --accent:     #22D3B8;
    --accent-dim: #14806F;
    --warn:       #F5A524;
    --danger:     #F0555A;
    --text-main:  #E7EDF3;
    --text-dim:   #7E8FA6;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 500px at 15% -10%, rgba(34,211,184,0.08), transparent 60%),
        var(--bg-deep);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--line);
}

/* Header */
.pa-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 1.5rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: linear-gradient(120deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
    margin-bottom: 1.2rem;
}
.pa-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-main);
    margin: 0;
}
.pa-title span { color: var(--accent); }
.pa-subtitle {
    color: var(--text-dim);
    font-size: 0.88rem;
    margin-top: 2px;
}
.pa-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    background: rgba(34,211,184,0.08);
    padding: 4px 10px;
    border-radius: 999px;
}

/* Agent cards */
.agent-card {
    border: 1px solid var(--line);
    background: var(--bg-panel);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    height: 100%;
}
.agent-card h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: var(--text-main);
}
.agent-card .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.agent-card p {
    color: var(--text-dim);
    font-size: 0.82rem;
    margin: 6px 0 0 0;
    line-height: 1.4;
}

/* Metric strip */
div[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] { color: var(--text-dim); }

/* Log console */
.log-console {
    background: #060A0F;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.55;
}
.log-info    { color: #8FE3D6; }
.log-warning { color: var(--warn); }
.log-safety  { color: #79A6FF; }
.log-success { color: var(--accent); font-weight: 600; }

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: #06110E;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
}
.stButton > button:hover {
    background: #38E4CB;
    color: #06110E;
}

hr { border-color: var(--line); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# AGENT CLASSES (adapted from the original notebook)
# ============================================================================


class PhysicalAgent:
    """Base class for physical AI agents."""

    def __init__(self, agent_id, name, agent_type):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.status = "idle"
        self.memory = deque(maxlen=200)
        self.safety_constraints = []

    def log(self, message, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"time": ts, "agent": self.agent_id, "level": level, "message": message}
        self.memory.append(entry)
        st.session_state.logs.append(entry)

    def add_safety_constraint(self, constraint):
        self.safety_constraints.append(constraint)
        self.log(f"Safety constraint registered", "SAFETY")

    def verify_safety(self, action):
        for constraint in self.safety_constraints:
            if not constraint(action):
                self.log("Safety violation detected — action blocked", "WARNING")
                return False
        return True


class VisionAgent(PhysicalAgent):
    """Simulated scene-understanding agent (CLIP-style zero-shot detection,
    approximated here with a lightweight stand-in so the app stays fast and
    dependency-light for cloud deployment)."""

    def __init__(self):
        super().__init__("VA-001", "Vision Agent", "VLM")
        self.log("Vision-language scene encoder ready")

    def detect_objects(self, seed=0):
        rng = np.random.default_rng(seed)
        queries = ["a red block", "a blue block", "a green block",
                   "a robotic arm", "an obstacle", "empty space"]
        base = rng.dirichlet(np.ones(len(queries)) * 1.5)
        results = {q: float(p) for q, p in zip(queries, base)}
        self.log(f"Scene scanned — {len(queries)} candidate concepts scored")
        for q, p in results.items():
            self.log(f"  '{q}': {p:.3f}")
        detected = [q for q, p in results.items() if p > 0.18]
        self.log(f"Detected {len(detected)} object(s) in frame")
        return detected, results


class PlanningAgent(PhysicalAgent):
    """LLM-style task decomposition agent with safety-gated actions."""

    def __init__(self):
        super().__init__("PA-001", "Planning Agent", "Reasoning")
        self.add_safety_constraint(lambda a: a.get("force", 0) < 10.0)
        self.add_safety_constraint(lambda a: a.get("speed", 0) < 5.0)
        self.add_safety_constraint(lambda a: "position" in a)

    def create_plan(self, task):
        self.log(f"Decomposing task: '{task}'")
        t = task.lower()
        if "pick" in t and "place" in t:
            plan = [
                {"action": "move_to", "position": [0.5, 0.5, 0.1], "speed": 2.0},
                {"action": "grasp", "force": 5.0, "position": [0.5, 0.5, 0.1]},
                {"action": "lift", "position": [0.5, 0.5, 0.4], "speed": 1.5},
                {"action": "move_to", "position": [0.8, 0.3, 0.3], "speed": 2.0},
                {"action": "release", "force": 0.0, "position": [0.8, 0.3, 0.3]},
                {"action": "return_home", "position": [0.0, 0.0, 0.5], "speed": 2.5},
            ]
        elif "avoid" in t:
            plan = [
                {"action": "scan_environment", "position": [0, 0, 0.5], "speed": 0.5},
                {"action": "calculate_path", "position": [0.4, 0.4, 0.3], "speed": 1.0},
                {"action": "move_safe", "position": [0.7, 0.7, 0.2], "speed": 1.0},
            ]
        else:
            plan = [{"action": "explore", "position": [0.5, 0.5, 0.3], "speed": 1.5}]

        self.log(f"Draft plan generated with {len(plan)} step(s)")
        safe_plan = []
        for a in plan:
            if self.verify_safety(a):
                safe_plan.append(a)
                self.log(f"  ✓ approved: {a['action']}")
            else:
                self.log(f"  ✗ removed: {a['action']}", "WARNING")
        return safe_plan


class NeuralController(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class ControlAgent(PhysicalAgent):
    """PyTorch neural controller mapping state → bounded action."""

    def __init__(self, state_dim=6, action_dim=3, seed=0):
        super().__init__("CA-001", "Control Agent", "PyTorch")
        torch.manual_seed(seed)
        self.controller = NeuralController(state_dim, action_dim)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=1e-3)
        self.log(f"Neural controller initialized ({state_dim}D → {action_dim}D)")

    def compute_action(self, state):
        with torch.no_grad():
            action = self.controller(torch.FloatTensor(state).unsqueeze(0))
        return action.squeeze().numpy()


class SafetyMonitorAgent(PhysicalAgent):
    """Lightweight safety classifier (small torch MLP standing in for the
    original TensorFlow model, so the app needs only one deep-learning
    backend for deployment)."""

    def __init__(self, seed=1):
        super().__init__("SM-001", "Safety Monitor", "Torch")
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.threshold = 0.55
        self.log("Safety classifier initialized")

    def assess_safety(self, state_action):
        features = np.array(state_action, dtype=np.float32).flatten()
        if len(features) < 9:
            features = np.pad(features, (0, 9 - len(features)))
        else:
            features = features[:9]
        with torch.no_grad():
            score = self.model(torch.FloatTensor(features).unsqueeze(0)).item()
        # bias score toward "safe" so the demo plan generally succeeds,
        # while still leaving room for occasional rejections
        score = 0.35 + 0.5 * score
        return score > self.threshold, score


class PhysicalEnvironment:
    """Simplified physics-free environment for the demo trajectory."""

    def __init__(self, target=(0.8, 0.3, 0.3), obstacle=(0.4, 0.4, 0.2, 0.1)):
        self.state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        self.target = np.array(target)
        self.obstacle_pos = np.array(obstacle[:3])
        self.obstacle_r = obstacle[3]

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        return self.state.copy()

    def step(self, action):
        self.state[:3] += action * 0.08
        dist = np.linalg.norm(self.state[:3] - self.target)
        reward = -dist
        collision = np.linalg.norm(self.state[:3] - self.obstacle_pos) < self.obstacle_r
        if collision:
            reward -= 10.0
        done = dist < 0.08
        return self.state.copy(), reward, done, collision


# ============================================================================
# SESSION STATE
# ============================================================================

if "logs" not in st.session_state:
    st.session_state.logs = []
if "results" not in st.session_state:
    st.session_state.results = None
if "has_run" not in st.session_state:
    st.session_state.has_run = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    """
    <div class="pa-header">
        <div>
            <p class="pa-title">🛡️ Safe<span>AI</span>-Agents</p>
            <p class="pa-subtitle">Multi-agent system for robotic manipulation, planning & safety verification</p>
        </div>
        <div class="pa-badge">VISION · PLANNING · CONTROL · SAFETY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Task Configuration")
    task_choice = st.selectbox(
        "Task",
        [
            "Pick and place the object while avoiding obstacles",
            "Avoid the obstacle and scan the area",
            "Explore the workspace",
        ],
    )
    num_steps = st.slider("Simulation steps", 10, 100, 50, step=10)
    seed = st.slider("Random seed", 0, 100, 7)

    st.markdown("---")
    st.markdown("### 🧠 Agents in this run")
    st.caption("Vision · Planning · Control · Safety Monitor")

    run_clicked = st.button("▶ Run Simulation", use_container_width=True)
    if st.session_state.has_run:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.logs = []
            st.session_state.results = None
            st.session_state.has_run = False
            st.rerun()

    st.markdown("---")
    st.caption("Adapted from a Colab research notebook targeting agentic, "
               "vision-language, and safe-AI robotics research.")

# ============================================================================
# SIMULATION RUNNER
# ============================================================================


def run_simulation(task, steps, seed):
    st.session_state.logs = []

    vision_agent = VisionAgent()
    planning_agent = PlanningAgent()
    control_agent = ControlAgent(seed=seed)
    safety_agent = SafetyMonitorAgent(seed=seed + 1)
    env = PhysicalEnvironment()

    detected_objects, vision_scores = vision_agent.detect_objects(seed=seed)
    plan = planning_agent.create_plan(task)

    state = env.reset()
    trajectory = [state[:3].copy()]
    distances = []
    rewards = []
    rejections = 0
    collisions = 0

    for step in range(steps):
        action = control_agent.compute_action(state)
        state_action = np.concatenate([state[:6], action])
        is_safe, score = safety_agent.assess_safety(state_action)

        if not is_safe:
            rejections += 1
            action = np.zeros(3)

        next_state, reward, done, collision = env.step(action)
        if collision:
            collisions += 1

        trajectory.append(next_state[:3].copy())
        distances.append(np.linalg.norm(next_state[:3] - env.target))
        rewards.append(reward)
        state = next_state

        if (step + 1) % 10 == 0 or done:
            control_agent.log(
                f"Step {step + 1}/{steps} — reward {reward:.3f}, "
                f"distance {distances[-1]:.3f}"
            )
        if done:
            control_agent.log("Target reached", "SUCCESS")
            break

    return {
        "plan": plan,
        "trajectory": np.array(trajectory),
        "distances": distances,
        "rewards": rewards,
        "total_reward": float(np.sum(rewards)),
        "detected_objects": detected_objects,
        "vision_scores": vision_scores,
        "rejections": rejections,
        "collisions": collisions,
        "target": env.target,
        "obstacle": env.obstacle_pos,
        "obstacle_r": env.obstacle_r,
        "steps_taken": len(trajectory) - 1,
    }


if run_clicked:
    with st.spinner("Running multi-agent pipeline…"):
        st.session_state.results = run_simulation(task_choice, num_steps, seed)
        st.session_state.has_run = True

# ============================================================================
# TABS
# ============================================================================

tab_overview, tab_run, tab_logs = st.tabs(["🏗️ Architecture", "📊 Simulation", "🖥️ Agent Console"])

# ---- Architecture tab ----
with tab_overview:
    st.markdown("#### How the four agents work together")
    cols = st.columns(4)
    agent_info = [
        ("👁️", "Vision Agent", "VLM", "Scores candidate scene concepts (objects, obstacles, free space) from the current frame."),
        ("🧭", "Planning Agent", "Reasoning", "Turns a natural-language task into a step-by-step action plan, filtering out anything that breaks a safety rule."),
        ("🎛️", "Control Agent", "PyTorch", "A small neural network maps the robot's state to a bounded control action at every step."),
        ("🛡️", "Safety Monitor", "Classifier", "Independently scores each state-action pair and can veto an action in real time."),
    ]
    for col, (icon, name, tag, desc) in zip(cols, agent_info):
        col.markdown(
            f"""
            <div class="agent-card">
                <div class="tag">{tag}</div>
                <h4>{icon} {name}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="agent-card">
            <h4>Pipeline</h4>
            <p>1. <b>Vision</b> scans the scene → 2. <b>Planning</b> drafts and safety-filters a plan →
            3. <b>Control</b> computes an action at each step → 4. <b>Safety Monitor</b> approves or
            rejects it before it reaches the environment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Configure a task in the sidebar and click **Run Simulation** to see it in action.")

# ---- Simulation tab ----
with tab_run:
    if not st.session_state.has_run:
        st.warning("No simulation run yet — use the sidebar to run one.")
    else:
        r = st.session_state.results

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reward", f"{r['total_reward']:.2f}")
        m2.metric("Steps taken", r["steps_taken"])
        m3.metric("Safety rejections", r["rejections"])
        m4.metric("Collisions", r["collisions"])

        st.markdown("#### Robot trajectory")
        traj = r["trajectory"]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines", line=dict(color="#22D3B8", width=5), name="Path",
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode="markers", marker=dict(size=6, color="#79A6FF"), name="Start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[r["target"][0]], y=[r["target"][1]], z=[r["target"][2]],
            mode="markers", marker=dict(size=8, color="#22D3B8", symbol="diamond"), name="Target",
        ))
        fig.add_trace(go.Scatter3d(
            x=[r["obstacle"][0]], y=[r["obstacle"][1]], z=[r["obstacle"][2]],
            mode="markers", marker=dict(size=8, color="#F0555A", symbol="x"), name="Obstacle",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="X", backgroundcolor="#0B0F14", gridcolor="#223047"),
                yaxis=dict(title="Y", backgroundcolor="#0B0F14", gridcolor="#223047"),
                zaxis=dict(title="Z", backgroundcolor="#0B0F14", gridcolor="#223047"),
            ),
            font=dict(color="#E7EDF3"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distance to target")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=r["distances"], mode="lines",
                                       line=dict(color="#22D3B8", width=2), name="Distance"))
            fig2.add_hline(y=0.08, line_dash="dash", line_color="#F0555A")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E7EDF3"), height=300,
                xaxis=dict(title="Step", gridcolor="#223047"),
                yaxis=dict(title="Distance", gridcolor="#223047"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("#### Scene detection confidence")
            labels = list(r["vision_scores"].keys())
            values = list(r["vision_scores"].values())
            fig3 = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color="#22D3B8",
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E7EDF3"), height=300,
                xaxis=dict(title="Confidence", range=[0, 1], gridcolor="#223047"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Approved action plan")
        st.table(r["plan"])

# ---- Console tab ----
with tab_logs:
    st.markdown("#### Live agent console")
    if not st.session_state.logs:
        st.caption("Logs will appear here once a simulation is run.")
    else:
        level_class = {
            "INFO": "log-info",
            "WARNING": "log-warning",
            "SAFETY": "log-safety",
            "SUCCESS": "log-success",
        }
        lines = []
        for entry in st.session_state.logs:
            cls = level_class.get(entry["level"], "log-info")
            lines.append(
                f'<div class="{cls}">[{entry["time"]}] [{entry["agent"]}] '
                f'{entry["level"]}: {entry["message"]}</div>'
            )
        st.markdown(f'<div class="log-console">{"".join(lines)}</div>', unsafe_allow_html=True)