import torch, torch.nn as nn, torch.nn.functional as F

class StepCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,16, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64, kernel_size=9, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64,1)
    def forward(self, x):        # x: (B,C,T) -> prob step at window end
        h = self.net(x).squeeze(-1)
        return torch.sigmoid(self.fc(h))

model = StepCNN(in_ch=1).eval()
# ... lade Daten (Fenster T=128 von |a| oder (x,y,z)), trainiere kurz ...
# Export:
ex = torch.randn(1,1,128)
traced = torch.jit.trace(model, ex)
traced = torch.jit.optimize_for_inference(traced)  # mobile opts
traced._save_for_lite_interpreter("export/step_cnn.ptl")
