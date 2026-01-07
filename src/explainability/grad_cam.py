import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM implementation for CNN- based models

    -  Computes class-discriminative localization maps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx = None):
        """
        Generate Grad-CAM heatmap for a single image.

        - input_tesnsor :  Input image tensor of chape (1, C, H,W)
        - class_idx : Target class index. If "None", uses the predicted class
        """

        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim = (2, 3), keepdim = True)

        cam = (weights * self.activations).sum(dim = 1)
        cam = F.relu(cam)

        # Normalization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().squeeze().cpu().numpy()
