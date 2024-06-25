import torch
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else device

class SmoothGrad():
    def __init__ (self, model, load, preprocess, saliency):
        self.model = model
        self.load = load
        self.preprocess = preprocess
        self.sal = saliency(model, load, preprocess)

    def smoothsaliency (self,  x_values_paths, prediction_class, noise_steps = 50, noise_var = 0.1, steps=20, steps_at=None, batch_size=32, x_values = None, **kwargs):
        if x_values is None:
            x_values = self.load(x_values_paths)
        elif isinstance(x_values, np.ndarray):
            x_values = self.load(x_values, False)
        noise_values = torch.Tensor(np.array([x_values]*noise_steps))
        noise_values = noise_values + torch.randn_like(noise_values)*np.sqrt(noise_var)
        noise_values *= 255
        if steps_at is None:
            steps_at = [steps]
        smoothn = np.zeros((len(steps_at), x_values.shape[0], x_values.shape[2], x_values.shape[3], x_values.shape[1]))
        smoothidgin = np.zeros((len(steps_at), x_values.shape[0], x_values.shape[2], x_values.shape[3], x_values.shape[1]))
        for i in range(noise_steps):
            tempn, tempidgin = self.sal.saliency(x_values_paths = x_values_paths, x_values = noise_values[i], prediction_class = prediction_class, steps = steps, steps_at = steps_at, batch_size=batch_size, **kwargs)
            smoothn += tempn
            smoothidgin += tempidgin
        return smoothn/noise_steps, smoothidgin/noise_steps

    def smoothvg(self,  x_values_paths, prediction_class, noise_steps = 50, noise_var = 0.1, batch_size=32, x_values = None):
        if x_values is None:
            x_values = self.load(x_values_paths)
        elif isinstance(x_values, np.ndarray):
            x_values = self.load(x_values, False)
        noise_values = torch.Tensor(np.array([x_values]*noise_steps))
        noise_values = noise_values + torch.randn_like(noise_values)*np.sqrt(noise_var)
        noise_values *= 255

        smoothvgn = np.zeros((x_values.shape[0], x_values.shape[2], x_values.shape[3], x_values.shape[1]))
        for i in range(noise_steps):
            tempn = self.sal.saliency(x_values_paths = x_values_paths, x_values = noise_values[i], prediction_class = prediction_class, batch_size = batch_size)
            smoothvgn += tempn
        return smoothvgn/noise_steps
