import torch
import torch.nn.functional as F
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)          
        self.stride = stride                          
        self.padding = padding   
        k = in_channels * kernel_size[0] * kernel_size[1]
        self.kernel = nn.Parameter(torch.randn(k, out_channels))



    def forward(self, input):                          
        padding, stride = self.padding, self.stride   
        batch_size, in_channels, height, width = input.size()
        assert in_channels == self.in_channels, f'expected {self.in_channels} channels, got {in_channels}'

        oh = (height + 2 * padding - self.kernel_size[0]) // stride + 1
        ow = (width + 2 * padding - self.kernel_size[1]) // stride + 1

        x = F.unfold(input, self.kernel_size, stride=stride, padding=padding)   
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        assert x.size() == (batch_size, k, oh * ow), f'expected {(batch_size, k, oh * ow)}, got {x.size()}' 
        Yp = x.transpose(1, 2) @ self.kernel

        Y = Yp.transpose(1, 2).reshape(batch_size, self.out_channels, oh, ow)
        assert Y.size() == (batch_size, self.out_channels, oh, ow)
        return Y

class Conv2DFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, kernel, stride=1, padding=1):
            b, c, h, w = input.size()
            c_out, c_in, kh, kw = kernel.size()
            assert c == c_in, f'expected {c_in} channels, got {c}'
            ctx.save_for_backward(input, kernel)
            ctx.stride = stride
            ctx.padding = padding
            oh = (h + 2 * padding - kh) // stride + 1
            ow = (w + 2 * padding - kw) // stride + 1
            W = kernel.reshape(c_out, -1).t()  # (k, c_out)
            U = F.unfold(input, (kh, kw), padding=padding, stride=stride)  
            Yp = U.transpose(1, 2) @ W                                           
            output_batch = Yp.transpose(1, 2).reshape(b, c_out, oh, ow)         
            return output_batch

        @staticmethod
        def backward(ctx, grad_output):
            input, kernel = ctx.saved_tensors
            stride, padding = ctx.stride, ctx.padding
            b, c, h, w = input.size()
            c_out, c_in, kh, kw = kernel.size()

            W = kernel.reshape(c_out, -1).t()                                    
            U = F.unfold(input, (kh, kw), padding=padding, stride=stride)  
            Yp_grad = grad_output.reshape(b, c_out, -1).transpose(1, 2)          
            W_grad = torch.einsum('bkp,bpo->ko', U, Yp_grad)                     
            kernel_grad = W_grad.t().reshape(c_out, c_in, kh, kw)
            U_grad = (Yp_grad @ W.t()).transpose(1, 2)                           
            input_batch_grad = F.fold(U_grad, output_size=(h, w), kernel_size=(kh, kw), padding=padding, stride=stride)            
            return input_batch_grad, kernel_grad, None, None
        
torch.manual_seed(0)
for s, p in [(1, 1), (2, 1), (2, 0)]:
    x = torch.randn(2, 3, 7, 7, dtype=torch.double, requires_grad=True)
    k = torch.randn(4, 3, 3, 3, dtype=torch.double, requires_grad=True)
    print(s, p,
          torch.allclose(Conv2DFunc.apply(x, k, s, p), F.conv2d(x, k, stride=s, padding=p)),
          torch.autograd.gradcheck(lambda a, b: Conv2DFunc.apply(a, b, s, p), (x, k)))