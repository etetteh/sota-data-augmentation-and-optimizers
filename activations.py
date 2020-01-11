from torch import nn
import torch
import torch.nn.functional as F


'''
Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
See additional documentation for mish class.
'''

'''    
class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        s_i = F.softplus(i)
        e_i = torch.exp(i)
        return grad_output * (torch.tanh(s_i) + (i * e_i * (F.math.acosh(s_i))**2) / (e_i + 1))


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)
'''


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
