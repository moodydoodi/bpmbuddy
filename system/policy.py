"""
AI Decision Policy Module.
Encapsulates Neural Network architectures (CNN, DQN) and provides an interface 
for model loading and decision making.
"""
import os
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import config as cfg


class ActivityCNN(nn.Module):
    """
    1D-Convolutional Neural Network for physical activity classification
    based on 3-axis accelerometer timeseries data.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=32 * 13, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DQN(nn.Module):
    """
    Deep Q-Network for Reinforcement Learning based strategy selection.
    Evaluates optimal music adjustments (keep, speed up, slow down, change energy).
    """
    def __init__(self, state_size: int = 4, action_size: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features=state_size, out_features=64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fully connected layers."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DecisionPolicy:
    """
    Wrapper class managing the execution of CNN perception and DQN strategy algorithms.
    """
    def __init__(self, use_rl: bool = True):
        self.device = torch.device("cpu")
        self.use_rl = use_rl
        self.cnn, self.dqn, self.classes = self._load_models()

    def _load_models(self) -> Tuple[Optional[ActivityCNN], Optional[DQN], List[str]]:
        """Loads model weights from disk safely."""
        classes = ["Jogging", "Sitting", "Standing", "Walking"]
        
        if not os.path.exists(cfg.MODEL_CNN_PATH):
            print("⚠️ CNN model missing. Using heuristic fallback.")
            return None, None, classes

        cnn = ActivityCNN(num_classes=len(classes)).to(self.device)
        dqn = DQN(state_size=4, action_size=4).to(self.device)

        try:
            c_state = torch.load(cfg.MODEL_CNN_PATH, map_location=self.device)
            if "model_state_dict" in c_state:
                cnn.load_state_dict(c_state["model_state_dict"])
            else:
                cnn.load_state_dict(c_state)

            if os.path.exists(cfg.MODEL_RL_PATH):
                dqn.load_state_dict(torch.load(cfg.MODEL_RL_PATH, map_location=self.device))
                
            cnn.eval()
            dqn.eval()
            print("🧠 AI models loaded successfully.")
            return cnn, dqn, classes
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return None, None, classes