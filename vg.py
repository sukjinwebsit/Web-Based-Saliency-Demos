import gc
import torch
import numpy as np

device = torch.device("cpu")

class VG():
    def __init__(self, model, load, preprocess):
        self.model = model
        self.load = load
        self.preprocess = preprocess
    def saliency(self, x_values_paths, prediction_class, batch_size=32, x_values = None):
        if x_values is None:
            x_values = self.load(x_values_paths)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        elif isinstance(x_values, np.ndarray):
            x_values = self.load(x_values, False)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        vgn = torch.zeros((x_values.shape[0], x_values.shape[1], x_values.shape[2], x_values.shape[3]))
        for k in range(0,x_values.shape[0],batch_size):
            k1 = k*batch_size
            k2 = np.min([(k+1)*batch_size, x_values.shape[0]])
            if (k1 >= k2):
                break
            x_value = x_values[k1:k2]
            x_batch = x_value
            x_batch = x_batch/255
            processed = self.preprocess(x_batch).to(device)
            target_class_idx = prediction_class[k1:k2]
            output = self.model(processed)
            m = torch.nn.Softmax(dim=1)
            output = m(output)
            outputs = output[torch.arange(x_value.shape[0]),target_class_idx]
            gradientsf = torch.autograd.grad(outputs, processed, grad_outputs=torch.ones_like(outputs))[0].detach()
            vgn[k1:k2] = gradientsf
            del gradientsf, outputs, x_batch, processed, output
            torch.mps.empty_cache()
            gc.collect()
        vgn = torch.movedim(vgn, (1,2,3),(3,1,2))
        return vgn.numpy()
