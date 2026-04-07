import torch
import torch.nn as nn

class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256),
            DCGAN_Generator.block(256, 128),
            DCGAN_Generator.block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)

# load weights
model = DCGAN_Generator()
model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=False))
model.eval()

# export
dummy = torch.randn(1, 100)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["noise"],
    output_names=["image"],
    opset_version=11,
    dynamo=False
)

print("Done! model.onnx created.")