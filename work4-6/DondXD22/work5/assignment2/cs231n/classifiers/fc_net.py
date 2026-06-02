from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        ############################################################################
        # TODO: 初始化两层网络的权重和偏置。权重应从均值为0.0、
        # 标准差等于weight_scale的正态分布中初始化，偏置应初始化为0。
        # 所有权重和偏置应存储在字典self.params中，
        # 第一层的权重和偏置使用键'W1'和'b1'，
        # 第二层的权重和偏置使用键'W2'和'b2'。#
        ############################################################################

        self.params["W1"]=np.random.randn(input_dim,hidden_dim)*weight_scale
        self.params["b1"]=np.zeros(hidden_dim)
        self.params["W2"]=np.random.randn(hidden_dim,num_classes)*weight_scale
        self.params["b2"]=np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X:np.ndarray, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        # TODO：实现双层网络的正向传播，计算X的类别得分并存储在scores变量中。

        y1,cache1=affine_relu_forward(X,self.params["W1"],self.params["b1"])
        scores,cache2=affine_forward(y1,self.params["W2"],self.params["b2"])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################


        # TODO: 实现两层网络的反向传播。将损失存储在loss变量中，
        # 梯度存储在grads字典中。使用softmax计算数据损失，
        # 并确保grads[k]包含self.params[k]的梯度。不要忘记添加L2正则化！
        # 注意：为了确保你的实现与我们的实现一致，并通过自动测试，
        # 请确保你的L2正则化包含一个0.5的因子，以简化梯度的表达式。

        loss,dx=softmax_loss(scores,y)
        loss+=0.5*np.sum(np.square(self.params["W1"]))*self.reg
        loss+=0.5*np.sum(np.square(self.params["W2"]))*self.reg
        dx2,dw2,db2=affine_backward(dx,cache2)
        grads["W2"]=dw2+self.params["W2"]*self.reg
        grads["b2"]=db2
        dx1,dw1,db1=affine_relu_backward(dx2,cache1)
        grads["W1"]=dw1+self.params["W1"]*self.reg
        grads["b1"]=db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """
    Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """
    """
    类：多层全连接神经网络

    该网络包含任意数量的隐藏层、ReLU非线性函数和softmax损失函数。
    此实现还将实现dropout和批/层归一化作为可选功能。
    对于一个具有L层的网络，其架构将是

    {仿射层 - [批/层归一化] - ReLU - [dropout]} x (L - 1) - 仿射层 - softmax

    其中批/层归一化和dropout是可选的，而{...}块重复L - 1次。

    可学习的参数存储在self.params字典中，并将使用Solver类进行学习。
    """

    def __init__(
        self,
        hidden_dims:list[int],
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        """
        初始化一个新的`FullyConnectedNet`。

        输入参数：
        - `hidden_dims`：一个整数列表，表示每个隐藏层的大小。
        - `input_dim`：一个整数，表示输入的大小。
        - `num_classes`：一个整数，表示分类的类别数。
        - `dropout_keep_ratio`：介于0和1之间的标量，表示dropout的强度。
            如果`dropout_keep_ratio=1`，则网络将不会使用dropout。
        - `normalization`：网络应使用的归一化类型。有效值有"batchnorm"（批归一化）、
            "layernorm"（层归一化）或`None`表示不使用归一化（默认）。
        - `reg`：一个标量，表示L2正则化的强度。
        - `weight_scale`：一个标量，表示权重随机初始化的标准差。
        - `dtype`：一个NumPy数据类型对象；所有计算都将使用此数据类型进行。
            `float32`更快但精度较低，因此您应该使用`float64`进行数值梯度检查。
        - `seed`：如果非`None`，则将此随机种子传递给dropout层。
            这将使dropout层具有确定性，因此我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        ############################################################################
        # TODO: 初始化网络的参数，将所有值存储在 self.params 字典中。
        # 将第一层的权重和偏置存储在 W1 和 b1 中；第二层使用 W2 和 b2，依此类推。
        # 权重应从以 0 为中心、标准差等于 weight_scale 的正态分布中初始化。偏置应初始化为零。
        # 当使用批量归一化时，将第一层的缩放参数和偏移参数存储在 gamma1 和 beta1 中；
        # 第二层使用 gamma2 和 beta2，依此类推。缩放参数应初始化为 1，偏移参数应初始化为 0。
        ############################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            cur=str(i+1)
            self.params["W"+cur]=np.random.randn(dims[i],dims[i+1])*weight_scale
            self.params["b"+cur]=np.zeros(dims[i+1])
            if self.normalization == 'batchnorm' and i!=self.num_layers-1:
               self.params["gamma"+cur]=np.ones(dims[i+1])
               self.params["beta"+cur]=np.zeros(dims[i+1])


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        ############################################################################
        # TODO: 实现 fully connected 网络的前向传播，计算 X 的类别得分并存储在 scores 变量中。#
        #                                                                          #
        # 在使用 dropout 时，需要将 self.dropout_param 传递给每个 dropout 前向传播。    #
        #                                                                          #
        # 在使用批标准化时，需要将 self.bn_params[0] 传递给第一个批标准化层的前向传播，#
        # 将 self.bn_params[1] 传递给第二个批标准化层的前向传播，依此类推。          #
        ############################################################################

        curr_input=X
        caches=[]
        for i in range(self.num_layers):
            if i == self.num_layers-1:
                curr_input,cache_i=affine_forward(curr_input,self.params[f'W{i+1}'],self.params[f'b{i+1}'])
            else:
                if self.normalization =='batchnorm':
                   curr_input,cache_i=affine_bn_relu_forward(curr_input,
                                                             self.params[f'W{i+1}'],self.params[f'b{i+1}'],
                                                             self.params[f'gamma{i+1}'],self.params[f'beta{i+1}'],self.bn_params[i])
                else:
                    curr_input,cache_i=affine_relu_forward(curr_input,self.params[f'W{i+1}'],self.params[f'b{i+1}'])
                if self.use_dropout:
                    curr_input,cache_dp=dropout_forward(curr_input,self.dropout_param)
                    cache_i=(cache_i,cache_dp)
            caches.append(cache_i)

        scores=curr_input
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        # TODO: 实现全连接网络的反向传播。将损失存储在 loss 变量中，
        # 将梯度存储在 grads 字典中。使用 softmax 计算数据损失，
        # 并确保 grads[k] 包含了 self.params[k] 的梯度。别忘了添加 L2 正则化！
        #
        # 注意：当使用批处理/层归一化时，您不需要对缩放和偏移参数进行正则化。
        #
        # 注意：为了确保您的实现与我们的实现一致并且通过自动测试，
        # 请确保您的 L2 正则化包含一个 0.5 的因子，以便简化梯度的表达式。
        ############################################################################

        dy=1
        for i in range(self.num_layers,0,-1):            
            W=f'W{i}'
            b=f'b{i}'
            gamma=f'gamma{i}'
            beta=f'beta{i}'

            if i == self.num_layers:
                loss,dy=softmax_loss(scores,y)
                dy,grads[W],grads[b]=affine_backward(dy,caches[i-1])
            else:
                if self.use_dropout:
                    cache_dp=caches[i-1][1]
                    dy=dropout_backward(dy,cache_dp)
                    caches[i-1]=caches[i-1][0]
                if self.normalization == 'batchnorm':
                    dy,grads[W],grads[b],grads[gamma],grads[beta]=affine_bn_relu_backward(dy,caches[i-1])
                else:
                    dy,grads[W],grads[b]=affine_relu_backward(dy,caches[i-1])
                

            loss+=0.5*np.sum(np.square(self.params[W]))*self.reg
            grads[W]+=self.params[W]*self.reg

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True