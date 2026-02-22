import cv2
import torch
import torch.nn as nn
import numpy as np

def get_module_by_path(model, path):
    cur = model
    for name in path.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur

def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        if isinstance(target_layer, str):
            self.target = get_module_by_path(model, target_layer)
        elif isinstance(target_layer, nn.Module):
            self.target = target_layer
        else:
            self.target = find_last_conv(model)
        self.handles = []
        self.activations = None
        self.gradients = None
        self.handles.append(self.target.register_forward_hook(self.fwd_hook))
        self.handles.append(self.target.register_full_backward_hook(self.bwd_hook))
    def fwd_hook(self, module, inp, out):
        self.activations = out.detach()
    def bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        loss = logits[torch.arange(x.size(0), device=logits.device), class_idx]
        loss = loss.sum()
        loss.backward(retain_graph=True)
        acts = self.activations
        grads = self.gradients
        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam = torch.relu(torch.sum(weights * acts, dim=1, keepdim=True))
        cam = cam - cam.amin(dim=(2,3), keepdim=True)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam

def cam_to_numpy(cam, size):
    cam = cam.squeeze(1)
    cams = []
    for i in range(cam.size(0)):
        c = cam[i].cpu().numpy()
        c = cv2.resize(c, size, interpolation=cv2.INTER_LINEAR)
        cams.append(c)
    return np.stack(cams, axis=0)

def overlay_heatmap(img, heat, alpha=0.4):
    hm = np.uint8(255 * heat)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    if img.max() <= 1.0:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.astype(np.uint8)
    out = cv2.addWeighted(hm, alpha, base, 1 - alpha, 0)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
