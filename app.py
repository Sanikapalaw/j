import streamlit as st
import numpy as np
import torch
import torch.nn as nn

st.set_page_config(layout="wide")

# ------------------ DQN MODEL ------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )

    def forward(self,x):
        return self.net(x)


# ------------------ AGENT ------------------

class Agent:

    def __init__(self):
        self.actions = [0,5,10,20]
        self.model = DQN(6,4)

    def get_q_values(self,state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.model(s).numpy()[0]
        return q


agent = Agent()


# ------------------ UI ------------------

st.title("Perishable Inventory Optimization — Interactive RL")

col1,col2 = st.columns([2,1])

# ---------- USER INPUT PANEL ----------
with col2:

    st.header("User Input")

    st.write("Inventory by freshness")

    d5 = st.number_input("Day 5 (Fresh)",0,50,10)
    d4 = st.number_input("Day 4",0,50,8)
    d3 = st.number_input("Day 3",0,50,4)
    d2 = st.number_input("Day 2",0,50,2)
    d1 = st.number_input("Day 1 (Critical)",0,50,1)

    demand = st.slider("Today's Demand",1,40,12)

    hold_cost = st.slider("Holding Cost",0.0,1.0,0.1)
    spoil_pen = st.slider("Spoilage Penalty",0.5,5.0,1.5)


# ---------- STATE CREATION ----------
state = np.array([
    d5/20, d4/20, d3/20, d2/20, d1/20,
    demand/50
])

q_vals = agent.get_q_values(state)
best_action = np.argmax(q_vals)

action_names = ["Hold","Order Small","Order Medium","Order Large"]


# ---------- INVENTORY SHELF ----------
with col1:

    st.subheader("Inventory Shelf")

    cols = st.columns(5)
    inv = [d5,d4,d3,d2,d1]

    for i,val in enumerate(inv):
        color = "🟢" if i<2 else "🟡" if i==2 else "🔴"
        cols[i].write(color * min(val,10))
        cols[i].caption(f"{val} units")


# ---------- Q VALUES PANEL ----------
st.subheader("Agent Decision Visualization")

qcols = st.columns(4)

for i in range(4):

    if i == best_action:
        qcols[i].success(
            f"{action_names[i]}\n\nQ = {q_vals[i]:.3f}"
        )
    else:
        qcols[i].info(
            f"{action_names[i]}\n\nQ = {q_vals[i]:.3f}"
        )


st.markdown("### ✅ Recommended Action")

st.success(action_names[best_action])
