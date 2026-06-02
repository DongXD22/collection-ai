from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        ############################################################################

        # TODO: 初始化三层卷积神经网络的权重和偏置。
        # 权重应从以0.0为中心、标准差等于weight_scale的高斯分布中初始化；
        # 偏置应初始化为零。所有权重和偏置应存储在字典self.params中。
        # 使用键'W1'和'b1'存储卷积层的权重和偏置；
        # 使用键'W2'和'b2'存储隐藏全连接层的权重和偏置，
        # 使用键'W3'和'b3'存储输出全连接层的权重和偏置。
        #
        # 重要提示：对于这次作业，你可以假设第一个卷积层的填充和步长已经选择，
        # 以**保持输入的宽度和高度不变**。查看loss()函数的开头，了解这是如何实现的。

        ############################################################################
        C,H,W=input_dim
        self.params['W1']=np.random.randn(num_filters,C,filter_size,filter_size)*weight_scale
        self.params['b1']=np.zeros((num_filters))

        self.params["W2"]=np.random.randn(num_filters*(H//2)*(W//2),hidden_dim)*weight_scale
        self.params["b2"]=np.zeros(hidden_dim)
        self.params["W3"]=np.random.randn(hidden_dim,num_classes)*weight_scale
        self.params["b3"]=np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # 
        ############################################################################
        # TODO: 实现三层卷积神经网络的正向传播，计算X的类别得分，
        # 并将它们存储在scores变量中。
        #
        # 注意：你可以使用在cs231n/fast_layers.py和
        # cs231n/layer_utils.py中定义的函数来实现你的实现（已导入）。
        ############################################################################
        h1,cache1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        h2,cache2=affine_relu_forward(h1,W2,b2)
        scores,cache3=affine_forward(h2,W3,b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,dx=softmax_loss(scores,y)

        loss+=0.5*np.sum(np.square(self.params["W3"]))*self.reg
        dx3,dw3,db3=affine_backward(dx,cache3)
        grads["W3"]=dw3+self.params["W3"]*self.reg
        grads["b3"]=db3

        loss+=0.5*np.sum(np.square(self.params["W2"]))*self.reg
        dx2,dw2,db2=affine_relu_backward(dx3,cache2)
        grads["W2"]=dw2+self.params["W2"]*self.reg
        grads["b2"]=db2

        loss+=0.5*np.sum(np.square(self.params["W1"]))*self.reg
        dx1,dw1,db1=conv_relu_pool_backward(dx2,cache1)
        grads["W1"]=dw1+self.params["W1"]*self.reg
        grads["b1"]=db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
