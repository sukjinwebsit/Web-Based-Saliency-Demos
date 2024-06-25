import gc
import torch
import numpy as np
from torchvision import transforms

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else device

class IG():
    def __init__(self, model, load, preprocess):
        self.model = model
        self.load = load
        self.preprocess = preprocess
    def saliency(self, x_values_paths, prediction_class, baseline = None, steps=20, steps_at=None, batch_size=32, x_values = None):
        if x_values is None:
            x_values = self.load(x_values_paths)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        elif isinstance(x_values, np.ndarray):
            x_values = self.load(x_values, False)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        if steps_at is None:
            steps_at = [steps]
        ign = torch.zeros((len(steps_at), x_values.shape[0], x_values.shape[1], x_values.shape[2], x_values.shape[3]))
        igidgin = torch.zeros((len(steps_at), x_values.shape[0], x_values.shape[1], x_values.shape[2], x_values.shape[3]))
        def reshape_fortran(x, shape):
            if len(x.shape) > 0:
                x = x.permute(*reversed(range(len(x.shape))))
            return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
        for k in range(0,x_values.shape[0],batch_size):
            k1 = k*batch_size
            k2 = np.min([(k+1)*batch_size, x_values.shape[0]])
            if (k1 >= k2):
                break
            x_value = x_values[k1:k2]
            x_step_batched = []
            if baseline is None or baseline == 'black':
                baseline = torch.zeros((x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            if baseline == 'white':
                baseline = torch.ones((x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            if baseline == 'random':
                baseline = torch.rand((x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            gradientsn = torch.zeros((steps, x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            outputsn = torch.zeros((steps, x_value.shape[0]))
            for i in range(steps):
                x_step = x_value + i*(baseline-x_value)/steps
                x_step_batched.append(x_step)
                if len(x_step_batched)*(x_value.shape[0]) >= batch_size or i == steps - 1:
                    x_step_batched = torch.stack(x_step_batched)
                    x_step_batch = torch.reshape(torch.swapaxes(x_step_batched, 0, 1), (x_step_batched.shape[0]*x_step_batched.shape[1], x_step_batched.shape[2], x_step_batched.shape[3], x_step_batched.shape[4]))
                    x_step_batch = x_step_batch/255
                    processed = self.preprocess(x_step_batch).to(device)
                    target_class_idx = np.repeat(prediction_class[k1:k2],x_step_batched.shape[0])
                    output = self.model(processed)
                    m = torch.nn.Softmax(dim=1)
                    output = m(output)
                    outputs = output[torch.arange(x_step_batch.shape[0]),target_class_idx]
                    gradientsf = torch.autograd.grad(outputs, processed, grad_outputs=torch.ones_like(outputs))[0].detach()
                    outputs = reshape_fortran(outputs,(x_step_batched.shape[0], x_step_batched.shape[1])).detach()
                    outputsn[i-x_step_batched.shape[0]+1:i+1] = outputs
                    gradientsf = reshape_fortran(gradientsf, (x_step_batched.shape[0], x_step_batched.shape[1], x_step_batched.shape[2], x_step_batched.shape[3], x_step_batched.shape[4]))
                    gradientsn[i-x_step_batched.shape[0]+1:i+1] = gradientsf
                    del gradientsf, outputs, x_step_batched, x_step_batch, processed, output
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    x_step_batched = []
            for i in range(steps):
                for j,k in enumerate(steps_at):
                    if (i%(steps//k) == 0 and i < steps-steps//k):
                        d = outputsn[i+steps//k]-outputsn[i]
                        d = torch.reshape(d, (d.shape[0], 1,1,1))
                        element_product = gradientsn[i]**2
                        tmpff = d*element_product
                        tmpfff = torch.sum(element_product, dim = [1,2,3], keepdim = True)
                        igidgin[j][k1:k2] +=tmpff/tmpfff
                    if (i%(steps//k) == 0):
                        tmp = gradientsn[i]
                        ign[j][k1:k2] += torch.multiply(tmp,((baseline-x_value)))
        ign = torch.movedim(ign, (2,3,4),(4,2,3))
        igidgin = torch.movedim(igidgin, (2,3,4),(4,2,3))
        return ign.numpy(), igidgin.numpy()
