from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x:np.ndarray, w:np.ndarray, b:np.ndarray):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out=x.reshape((x.shape[0],w.shape[0])).dot(w)+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout:np.ndarray, cache:tuple[np.ndarray]):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx=dout.dot(w.T).reshape(x.shape)
    dw=x.reshape((x.shape[0],-1)).T.dot(dout)
    db=dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x:np.ndarray):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out=np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx=(x>0)*dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x:np.ndarray, gamma:np.ndarray, 
                      beta:np.ndarray, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    """
        批量归一化前向传播。

        在训练过程中，样本均值和（未校正的）样本方差从批处理统计数据中计算得出，
        并用于归一化输入数据。在训练过程中，
        我们还保留每个特征的均值和方差的指数衰减运行均值，
        这些平均值用于测试时归一化数据。

        在每个时间步长，我们使用动量参数基于指数衰减更新均值和方差的运行平均值：

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        请注意，批量归一化论文建议不同的测试时间行为：他们使用大量训练图像而不是使用运行平均值来为每个特征计算样本均值和方差。对于此实现，我们选择使用运行平均值，因为它们不需要额外的估计步骤；torch7的批量归一化实现也使用运行平均值。

        输入：
        - x：形状为（N，D）的数据
        - gamma：形状为（D，）的缩放参数
        - beta：形状为（D，）的平移参数
        - bn_param：包含以下键的字典：
            - mode：'train'或'test'；必需
            - eps：用于数值稳定性的常数
            - momentum：用于运行均值/方差的常数
            - running_mean：形状为（D，）的数组，给出特征的运行均值
            - running_var：形状为（D，）的数组，给出特征的运行方差

        返回一个元组：
        - out：形状为（N，D）
        - cache：反向传播所需值的元组
    """

    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean=x.mean(axis=0)
        sample_var=x.var(axis=0)
        x_hat=(x-sample_mean)/np.sqrt(sample_var+eps)
        out=x_hat*gamma+beta

        running_mean=momentum*running_mean+(1-momentum)*x.mean(axis=0)
        running_var=momentum*running_var+(1-momentum)*x.var(axis=0)

        
        cache=(x,x_hat,gamma,beta,sample_mean,sample_var,eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        #######################################################################
        # TODO: 实现批归一化的测试时前向传播。
        # 使用运行均值和方差对输入数据进行归一化，
        # 然后使用伽马（gamma）和贝塔（beta）参数对归一化后的数据进行缩放和偏移。
        # 将结果存储在 out 变量中。
        #######################################################################

        x_hat=(x-running_mean)/np.sqrt(running_var+eps)
        out=x_hat*gamma+beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout:np.ndarray, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    """
    反向传播过程用于批标准化。

    对于这个实现，你应该在纸上画出批标准化的计算图，并反向传播梯度通过中间节点。

    输入：
    - dout：上游导数，形状为 (N, D)
    - cache：batchnorm_forward 中间结果的变量。

    返回一个元组：
    - dx：对输入 x 的梯度，形状为 (N, D)
    - dgamma：对缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta：对平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    x,x_hat,gamma,beta,sample_mean,sample_var,eps=cache
    N,D=dout.shape

    dbeta=dout.sum(axis=0)
    dgamma=(x_hat*dout).sum(axis=0)

    dx_hat=gamma*dout
    dvar=(dx_hat*(x-sample_mean)*(-1/2)*(sample_var+eps)**(-3/2)).sum(axis=0)
    dmean=np.sum(dx_hat*(-1/np.sqrt(sample_var+eps))+dvar*(-2*(x-sample_mean))/N,axis=0)
    dx=dx_hat*np.sqrt(1/(sample_var+eps))+dvar*2*(x-sample_mean)/N+dmean/N



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    ###########################################################################
    # TODO: 实现批归一化的反向传播。将结果存储在 dx、dgamma 和 dbeta 变量中。
    #                                                                         #
    # 在计算相对于中心化输入的梯度后，你应该能够通过一条语句计算相对于输入的梯度；
    # 我们的实现可以适应单行80个字符的长度。
    ###########################################################################
    x,x_hat,gamma,beta,sample_mean,sample_var,eps=cache
    N,D=dout.shape

    dbeta=dout.sum(axis=0)
    dgamma=(x_hat*dout).sum(axis=0)
    std_inv = 1.0 / np.sqrt(sample_var + eps)
    
    dx_hat=gamma*dout
    dx = (1.0 / N) * std_inv * (
        N * dx_hat 
        - np.sum(dx_hat, axis=0) 
        - x_hat * np.sum(dx_hat * x_hat, axis=0)
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    sample_mean=x.mean(axis=1,keepdims=True)
    sample_var=x.var(axis=1,keepdims=True)
    x_hat=(x-sample_mean)/np.sqrt(sample_var+eps)
    out=x_hat*gamma+beta
    cache=(x,x_hat,gamma,beta,sample_mean,sample_var,eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x,x_hat,gamma,beta,sample_mean,sample_var,eps=cache
    N,D=dout.shape

    dbeta=dout.sum(axis=0)
    dgamma=(x_hat*dout).sum(axis=0)
    std_inv = 1.0 / np.sqrt(sample_var + eps)
    
    dx_hat=gamma*dout
    dx = (1.0 / D) * std_inv * (
        D * dx_hat 
        - np.sum(dx_hat, axis=1,keepdims=True) 
        - x_hat * np.sum(dx_hat * x_hat, axis=1,keepdims=True)
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    """
执行（反向）dropout前向传播。

    输入：
    - x：输入数据，可以是任何形状
    - dropout_param：一个包含以下键的字典：
      - p：dropout参数。我们以概率p保留每个神经元的输出。
      - mode：'test'或'train'。如果模式是train，则执行dropout；
      如果模式是test，则仅返回输入。
      - seed：随机数生成器的种子。传递种子使此函数具有确定性，
      这对于梯度检查是必需的，但在真实网络中不是必需的。

    输出：
    - out：与x相同形状的数组。
    - cache：元组（dropout_param，mask）。在训练模式下，
    mask是用于乘以输入的dropout掩码；在测试模式下，mask为None。

    备注：请实现**反向**dropout，而不是普通的dropout版本。
    请参阅http://cs231n.github.io/neural-networks-2/#reg以获取更多详细信息。

    备注2：请注意，p是保留神经元输出的概率；
    这可能与某些资料中的描述相反，其中它被称为丢弃神经元输出的概率。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask=np.random.rand(*x.shape)<p
        out=x*mask/p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out=x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx=dout/dropout_param["p"]*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    """
    # 前向传播卷积层的朴素实现

    输入由 N 个数据点组成，每个数据点包含 C 个通道、高度 H 和宽度 W。我们使用 F 个不同的过滤器对每个输入进行卷积，每个过滤器跨越所有 C 个通道，高度为 HH，宽度为 WW。

    输入：
    - x：形状为 (N, C, H, W) 的输入数据
    - w：形状为 (F, C, HH, WW) 的过滤器权重
    - b：偏置，形状为 (F,)
    - conv_param：一个包含以下键的字典：
    - 'stride'：相邻感受野在水平和垂直方向上的像素数
    - 'pad'：用于输入的零填充像素数


    在填充期间，'pad' 零应在对称（即在输入的高度和宽度轴上均匀）地放置。
    请注意，不要直接修改原始输入 x。

    返回一个元组：
    - out：输出数据，形状为 (N, F, H', W')，其中 H' 和 W' 由以下公式给出
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache：`(x, w, b, conv_param)`
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride,pad=conv_param['stride'],conv_param['pad']

    
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride

    out=np.zeros((N,F,H_,W_))
    x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))

    for i in range(N):
        for j in range(F):
            for h_ in range(H_):
                for w_ in range(W_):
                    hx=h_*stride
                    wx=w_*stride
                    out[i,j,h_,w_]+=(x_pad[i,:,hx:hx+HH,wx:wx+WW]*w[j]).sum()+b[j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param=cache
    stride,pad=conv_param['stride'],conv_param['pad']
    x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))

    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    N,F,H_,W_=dout.shape
    dx,dw,db,dx_pad=np.zeros(x.shape),np.zeros(w.shape),np.zeros(b.shape),np.zeros(x_pad.shape)
    for i in range(N):
        for j in range(F):
            for h_ in range(H_):
                for w_ in range(W_):
                    hx=h_*stride
                    wx=w_*stride
                    dx_pad[i,:,hx:hx+HH,wx:wx+WW]+=dout[i,j,h_,w_]*w[j]
                    dw[j]+=dout[i,j,h_,w_]*x_pad[i,:,hx:hx+HH,wx:wx+WW]
                    db[j]+=dout[i,j,h_,w_]
    dx=dx_pad[:,:,pad:pad+H,pad:pad+W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height,pool_width,stride=pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    N,C,H,W=x.shape
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    out=np.zeros((N,C,H_,W_))
    for n in range(N):
        for c in range(C):
            for h_ in range(H_):
                for w_ in range(W_):
                    hx=h_*stride
                    wx=w_*stride
                    out[n,c,h_,w_]=np.max(x[n,c,hx:hx+pool_height,wx:wx+pool_width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param=cache
    pool_height,pool_width,stride=pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    N,C,H,W=x.shape
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    dx=np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for h_ in range(H_):
                for w_ in range(W_):
                    hx=h_*stride
                    wx=w_*stride
                    idx=np.argmax(x[n,c,hx:hx+pool_height,wx:wx+pool_width])
                    h,w=np.unravel_index(idx,(pool_height,pool_width))
                    dx[n,c,hx+h,wx+w]+=dout[n,c,h_,w_]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    """
    计算空间批量归一化的正向传播。

        输入：
        - x：输入数据，形状为 (N, C, H, W)
        - gamma：尺度参数，形状为 (C,)
        - beta：平移参数，形状为 (C,)
        - bn_param：包含以下键的字典：
            - mode：'train' 或 'test'；必需
            - eps：用于数值稳定的常数
            - momentum：用于运行均值/方差的常数。momentum=0 
            表示在每一步完全丢弃旧信息，而 momentum=1 表示新信息永远不会被整合。
            默认的 momentum=0.9 在大多数情况下应该表现良好。
            - running_mean：形状为 (D,) 的数组，给出特征的运行均值
            - running_var：形状为 (D,) 的数组，给出特征的运行方差

        返回一个元组：
        - out：输出数据，形状为 (N, C, H, W)
        - cache：反向传播所需值
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    ###########################################################################
    # TODO: 实现空间批量归一化的正向传播。
    #
    # HINT：您可以通过调用上面实现的原始批量归一化版本来实现空间批量归一化。
    # 您的实现应该非常简洁；我们的实现不到五行。
    ###########################################################################
    N,C,H,W=x.shape
    x_flat=x.transpose(0,2,3,1).reshape(-1,C)
    out,cache=batchnorm_forward(x_flat,gamma,beta,bn_param)
    out=np.reshape(out,(N,H,W,C)).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout:np.ndarray, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    ###########################################################################
    # TODO: 实现空间批量归一化的反向传播。
    #
    # 提示：你可以通过调用前面实现的标准批量归一化版本来执行空间批量归一化。
    # 你的实现应该非常简洁；我们的实现不到五行代码。
    ###########################################################################
    N,C,H,W=dout.shape
    dout_flat=dout.transpose(0,2,3,1).reshape(-1,C)
    dx,dgamma,dbeta=batchnorm_backward_alt(dout_flat,cache)
    dx=dx.reshape(N,H,W,C).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    """
    计算空间组归一化的前向传播。
    与层归一化不同，组归一化将数据中的每个条目分为 G 个连续的部分，
    然后对这些部分进行独立的归一化。
    随后，对数据进行特征平移和缩放，其方式与批归一化和层归一化相同。

        输入：
        - x: 形状为 (N, C, H, W) 的输入数据
        - gamma: 缩放参数，形状为 (1, C, 1, 1)
        - beta: 平移参数，形状为 (1, C, 1, 1)
        - G: 分组后的整数组数，应为 C 的约数
        - gn_param: 包含以下键的字典：
            

        返回一个包含以下内容的元组：
        - out: 输出数据，形状为 (N, C, H, W)
        - cache: 用于反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    ###########################################################################
    # TODO: 实现空间分组归一化的前向传播。
    # 这将与层归一化的实现非常相似。
    # 尤其要考虑如何变换矩阵，以便大部分代码可以同时适用于
    # 训练时的批量归一化和层归一化！    
    ###########################################################################
    N,C,H,W=x.shape
    x=np.reshape(x,(N,G,-1))
    sample_mean=x.mean(axis=2,keepdims=True)
    sample_var=x.var(axis=2,keepdims=True)
    x_hat=(x-sample_mean)/np.sqrt(sample_var+eps)
    x_hat=np.reshape(x_hat,(N,C,H,W))
    out=x_hat*gamma+beta
    cache=(G,x_hat,gamma,beta,sample_mean,sample_var,eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    G,x_hat,gamma,beta,sample_mean,sample_var,eps=cache
    N,C,H,W=dout.shape

    dbeta=dout.sum(axis=(0,2,3),keepdims=True)
    dgamma=(x_hat*dout).sum(axis=(0,2,3),keepdims=True)

    std_inv = 1.0 / np.sqrt(sample_var + eps)
    
    dx_hat=gamma*dout
    x_hat=np.reshape(x_hat,(N,G,-1))
    dx_hat=np.reshape(dx_hat,(N,G,-1))
    D=dx_hat.shape[2]
    dx = (1.0 / D) * std_inv * (
        D * dx_hat 
        - np.sum(dx_hat, axis=2,keepdims=True) 
        - x_hat * np.sum(dx_hat * x_hat, axis=2,keepdims=True)
    )
    dx=np.reshape(dx,(N,C,H,W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    num_train = x.shape[0]

    x_max=np.max(x,axis=1,keepdims=True)
    shifted_x=x-x_max

    exps=np.exp(shifted_x)
    exps_sum=np.sum(exps,axis=1,keepdims=True)
    p=exps/exps_sum
    logp=np.log(p)
    loss=-np.sum(logp[np.arange(num_train),y])


    p_=np.copy(p)
    p_[np.arange(num_train),y]-=1
    dx=p_

    loss/=num_train
    dx/=num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
