"""This file defines layer types that are commonly used for recurrent neural networks.
"""
import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x:torch.tensor, prev_h:torch.tensor, Wx:torch.tensor, Wh:torch.tensor, b:torch.tensor):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    """
    """
    运行一个vanilla RNN的单个时间步长的正向传播，使用tanh激活函数。

    输入数据维度为D，隐藏状态维度为H，小批量大小为N。

    输入：
    - x：本时间步长的输入数据，形状为(N, D)
    - prev_h：前一个时间步长的隐藏状态，形状为(N, H)
    - Wx：输入到隐藏层的权重矩阵，形状为(D, H)
    - Wh：隐藏层到隐藏层的权重矩阵，形状为(H, H)
    - b：偏置，形状为(H,)

    返回一个元组，包含：
    - next_h：下一个隐藏状态，形状为(N, H)
    """
    next_h = None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN.                 #
    ##############################################################################
    
    next_h=torch.tanh(prev_h@Wh+x@Wx+b)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h


def rnn_forward(x:torch.tensor, h0:torch.tensor, Wx:torch.tensor, Wh:torch.tensor, b:torch.tensor):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    """
    """
    在数据序列上运行一个标准RNN。

    我们假设一个由T个向量组成的输入序列，每个向量的维度为D。RNN使用隐藏大小为H，我们处理一个包含N个序列的小批量。在运行RNN之前，我们返回所有时间步的隐藏状态。

    输入：
    - x：整个时间序列的输入数据，形状为（N, T, D）
    - h0：初始隐藏状态，形状为（N, H）
    - Wx：输入到隐藏连接的权重矩阵，形状为（D, H）
    - Wh：隐藏到隐藏连接的权重矩阵，形状为（H, H）
    - b：偏置，形状为（H,）

    返回一个元组：
    - h：整个时间序列的隐藏状态，形状为（N, T, H）    """
    h = None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################

    ##############################################################################
    # TODO: 实现一个在输入数据序列上运行的简单循环神经网络
    # （RNN）的前向传播。您应该使用上面定义的 `rnn_step_forward` 函数。
    # 您可以使用循环来帮助计算前向传播。
    ##############################################################################

    N,T,D=x.shape
    _,H=h0.shape
    h=torch.zeros((N,T,H),dtype=x.dtype)
    prev_h=h0

    for i in range(T):
        xt = x[:, i, :]
        next_h=rnn_step_forward(xt,prev_h,Wx,Wh,b)
        h[:,i,:]=next_h
        prev_h=next_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    """
    前向传播用于词嵌入。

    我们在大小为N的迷你批次上操作，其中每个序列的长度为T。
    我们假设有一个V个单词的词汇表，将每个单词分配到D维度的向量。

    输入：
    - x：形状为(N, T)的整数数组，给出单词的索引。
    x中的每个元素idx必须在0 <= idx < V的范围内。
    - W：形状为(V, D)的权重矩阵，给出所有单词的词向量。

    返回一个包含以下内容的元组：
    - out：形状为(N, T, D)的数组，给出所有输入单词的词向量。    """
    out = None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using Pytorch's array indexing.         #
    ##############################################################################
    out=W[x]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    """
    LSTM单时间步前向传播。

    输入数据维度为D，隐藏状态维度为H，我们使用的小批量大小为N。

    注意：此文件中已为您提供了sigmoid()函数。

    输入：
    - x：输入数据，形状为(N, D)
    - prev_h：前一个隐藏状态，形状为(N, H)
    - prev_c：前一个细胞状态，形状为(N, H)
    - Wx：输入到隐藏权重，形状为(D, 4H)
    - Wh：隐藏到隐藏权重，形状为(H, 4H)
    - b：偏置，形状为(4H,)

    返回一个元组，包含：
    - next_h：下一个隐藏状态，形状为(N, H)
    - next_c：下一个细胞状态，形状为(N, H)    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
