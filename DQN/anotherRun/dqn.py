import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, 256)  # Another layer with more neurons
        self.fc3 = nn.Linear(256, 128)  # Additional layer
        self.fc4 = nn.Linear(128, action_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99999,
                 min_epsilon=0.15, memory_size=10000, batch_size=128, device='cpu', log_dir='runs'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = device
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

        # Set up TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self, episode):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TensorBoard logging
        self.writer.add_scalar('Loss', loss.item(), episode)
        self.writer.add_scalar('Epsilon', self.epsilon, episode)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    # Save the model and related components
    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, file_path)

    # Load the model and related components
    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    # Close the TensorBoard writer
    def close_writer(self):
        self.writer.flush()  # Ensure all data is written
        self.writer.close()
