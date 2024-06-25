import gc, math
import torch
import numpy as np
from torchvision import transforms

device = torch.device("cpu")

class BlurIG():
    def __init__(self, model, load, preprocess):
        self.model = model
        self.load = load
        self.preprocess = preprocess
    def saliency(self, x_values_paths, prediction_class, steps=20, steps_at=None, batch_size=32, max_sigma = 50, grad_step=0.01, sqrt=False, x_values = None):
        if x_values is None:
            x_values = self.load(x_values_paths)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        elif isinstance(x_values, np.ndarray):
            x_values = self.load(x_values, False)
            x_values = x_values*255
            x_values = torch.from_numpy(x_values)
        if sqrt:
            sigmas = [math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)]
        else:
            sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
        step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]
        if steps_at is None:
            steps_at = [steps]
        blurign = torch.zeros((len(steps_at), x_values.shape[0], x_values.shape[1], x_values.shape[2], x_values.shape[3]))
        blurigidgin = torch.zeros((len(steps_at), x_values.shape[0], x_values.shape[1], x_values.shape[2], x_values.shape[3]))
        def gaussian_blur(images, sigma):
            if sigma == 0:
                return images
            return transforms.GaussianBlur(101, sigma).forward(images)
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
            gaussian_gradient_batched = []
            gaussian_gradientn = torch.zeros((steps, x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            gradientsn = torch.zeros((steps, x_value.shape[0], x_value.shape[1], x_value.shape[2], x_value.shape[3]))
            outputsn = torch.zeros((steps, x_value.shape[0]))
            for i in range(steps):
                x_step = gaussian_blur(x_value, sigmas[i])
                gaussian_gradient = (gaussian_blur(x_value,sigmas[i]+grad_step)-x_step)/grad_step
                x_step_batched.append(x_step)
                gaussian_gradient_batched.append(gaussian_gradient)
                gaussian_gradientn[i] = gaussian_gradient
                if len(x_step_batched)*(x_value.shape[0]) >= batch_size or i == steps - 1:
                    x_step_batched = torch.stack(x_step_batched)
                    gaussian_gradient_batched = torch.stack(gaussian_gradient_batched)
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
                    torch.mps.empty_cache()
                    gc.collect()
                    x_step_batched = []
                    gaussian_gradient_batched = []
            for i in range(steps):
                for j,k in enumerate(steps_at):
                    if (i%(steps//k) == 0 and i < steps-steps//k):
                        d = outputsn[i+steps//k]-outputsn[i]
                        d = torch.reshape(d, (d.shape[0], 1,1,1))
                        element_product = gradientsn[i]**2
                        tmpff = d*element_product
                        tmpfff = torch.sum(element_product, dim = [1,2,3], keepdim = True)
                        blurigidgin[j][k1:k2] +=tmpff/tmpfff
                    if (i%(steps//k) == 0):
                        tmp = np.sum(step_vector_diff[i:i+steps//k])*torch.multiply(gaussian_gradientn[i],gradientsn[i])
                        blurign[j][k1:k2] += tmp
        blurign *= -1.0
        blurign = torch.movedim(blurign, (2,3,4),(4,2,3))
        blurigidgin = torch.movedim(blurigidgin, (2,3,4),(4,2,3))
        return blurign.numpy(), blurigidgin.numpy()
