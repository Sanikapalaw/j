import streamlit as st
from environment import PerishableInventoryEnv
from agent import Agent
from replay_buffer import ReplayBuffer

st.title("Perishable Inventory Optimization — Reinforcement Learning")

env = PerishableInventoryEnv()
state_dim = len(env.get_state())
action_dim = len(env.action_space)

agent = Agent(state_dim, action_dim)
buffer = ReplayBuffer()

episodes = st.slider("Episodes", 10, 200, 50)

def render_inventory(inventory, shelf_life):

    st.subheader("Inventory Shelf")

    age_dict = {i:0 for i in range(1, shelf_life+1)}

    for qty, age in inventory:
        age_dict[age] += qty

    cols = st.columns(shelf_life)

    for i, age in enumerate(range(shelf_life,0,-1)):

        qty = age_dict[age]

        if age >= 3:
            color = "🟢"
        elif age == 2:
            color = "🟡"
        else:
            color = "🔴"

        cols[i].markdown(f"**Day {age}**")
        cols[i].write(color * min(qty,10))
        cols[i].caption(f"{qty} units")


if st.button("Train Agent"):

    rewards = []

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0

        for step in range(30):

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            agent.train_step(buffer)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    st.line_chart(rewards)
    render_inventory(env.inventory, env.shelf_life)

    st.success("Training Complete")
