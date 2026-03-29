import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

def run_kmeans(X, n_clusters=2, random_state=42):
    """Runs a MiniBatchKMeans for faster performance."""
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1024)
    kmeans.fit(X)
    return kmeans

def train_xgboost(X_train, y_train):
    """Trains an XGBoost classifier."""
    # Scale positive class weight to handle imbalance (usually ratio of neg/pos)
    pos_count = max(y_train.sum(), 1)
    scale_pos_weight = (len(y_train) - pos_count) / pos_count
    
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb

def compute_shap_values(model, X_sample):
    """Computes SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

class SimpleQLearningAgent:
    """A tabular Q-Learning agent for prescriptive analytics."""
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {s: {a: 0.0 for a in actions} for s in states}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            action_values = self.q_table[state]
            return max(action_values, key=action_values.get)
            
    def update(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

def run_rl_simulation(y_true, y_prob):
    """
    Simulates an RL agent deciding to Auto-Approve, Review, or Deny.
    States are probability bins (0-10%, 10-20% ...).
    Rewards based on business logic.
    """
    # Create bins for states
    bins = np.linspace(0, 1, 11) # 10 bins
    states = [f"Bin_{i}" for i in range(10)]
    actions = ["Approve", "Review", "Deny"]
    
    agent = SimpleQLearningAgent(states, actions)
    
    # Simple simulation loop (1 epoch)
    # To avoid hanging, use a small sample 
    rewards_history = []
    
    # Zip true labels and their prediction probability
    data = list(zip(y_true, y_prob))
    import random
    random.shuffle(data)
    
    cumulative_reward = 0
    for true_label, prob in data:
        # Determine state
        bin_idx = min(int(prob * 10), 9)
        state = states[bin_idx]
        
        # Choose action
        action = agent.choose_action(state)
        
        # Calculate reward
        reward = 0
        if action == "Approve":
            if true_label == 1:
                reward = -100  # huge loss for approving fraud
            else:
                reward = 1     # normal transaction fee
        elif action == "Review":
            reward = -2        # cost of human review
            # Assuming review correctly catches it eventually, no further loss
        else: # Deny
            if true_label == 1:
                reward = 10    # saved loss
            else:
                reward = -5    # lost customer trust/fee
        
        agent.update(state, action, reward, state) # next_state is essentially same here (independent events)
        cumulative_reward += reward
        rewards_history.append(cumulative_reward)
        
    return agent.q_table, rewards_history
