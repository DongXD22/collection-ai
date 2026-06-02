import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # 将字幕分为两部分：captions_in 包含除最后一个词之外的所有内容
        # 并将作为 RNN 的输入；captions_out 包含除第一个词之外的所有内容
        # 这是我们期望 RNN 生成的结果。这两部分相对于彼此偏移了一个位置，
        # 因为 RNN 应该在接收到词 t 之后生成词 (t+1)。captions_in 的第一个元素将是 START
        # 标记，而 captions_out 的第一个元素将是第一个词。

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.                  #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # You also don't have to implement the backward pass.                      #
        ###########################################################################
        ############################################################################
        # TODO: 实现 CaptioningRNN 的前向传播。
        # 在前向传播过程中，您需要执行以下步骤：
        # (1) 使用仿射变换从图像特征计算初始隐藏状态，
        # 这应该生成一个形状为 (N, H) 的数组。
        # (2) 使用词嵌入层将 captions_in 中的单词从索引转换为向量，
        # 得到一个形状为 (N, T, W) 的数组。
        # (3) 使用标准 RNN 或 LSTM（取决于 self.cell_type）来处理输入单词向量的序列，
        # 并为所有时间步产生隐藏状态向量，生成一个形状为 (N, T, H) 的数组。
        # (4) 使用（时序）仿射变换计算每个时间步的词汇表分数，使用隐藏状态，
        # 得到一个形状为 (N, T, V) 的数组。
        # (5) 使用（时序）softmax 计算损失，使用 captions_out，
        # 忽略 mask 中的 <NULL> 输出点。
        # 请不要担心正则化权重或其梯度！
        # 您也不必实现反向传播。
        ############################################################################
        h0=affine_forward(features,W_proj,b_proj)
        x=word_embedding_forward(captions_in,W_embed)
        if self.cell_type == 'rnn':
          out=rnn_forward(x,h0,Wx,Wh,b)
        scores=temporal_affine_forward(out,W_vocab,b_vocab)
        loss=temporal_softmax_loss(scores,captions_out,mask)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        ###########################################################################
        # TODO: 实现模型的测试时采样。您需要通过应用学习到的仿射变换
        # 来初始化RNN的隐藏状态。您首先需要输入到RNN中的单词是<START>标记；
        # 其值存储在变量self._start中。在每个时间步，您需要执行以下操作：
        # (1) 使用学习到的词嵌入将前一个单词嵌入
        # (2) 使用前一个隐藏状态和嵌入的当前单词进行RNN步骤，
        # 以获取下一个隐藏状态
        # (3) 对下一个隐藏状态应用学习到的仿射变换，
        # 以获取词汇表中所有单词的分数
        # (4) 选择得分最高的单词作为下一个单词，
        # 将其（单词索引）写入captions变量中的适当槽位
        # 
        # 为了简化，您不需要在采样到<END>标记后停止生成，但如果您想这样做也可以。
        # 
        # 提示：您将无法使用rnn_forward或lstm_forward函数；
        # 您需要在一个循环中调用rnn_step_forward或lstm_step_forward。
        # 
        # 注意：在这个函数中，我们仍在处理小批量数据。另外，
        # 如果您使用LSTM，请将第一个细胞状态初始化为零。       
        ############################################################################
        N=features.shape[0]
        prev_h=affine_forward(features,W_proj,b_proj)
        idx=torch.full((N,),self._start,dtype=torch.long)
        for t in range(max_length):
          x=word_embedding_forward(idx,W_embed)
          next_h=rnn_step_forward(x,prev_h,Wx,Wh,b)
          scores=affine_forward(next_h,W_vocab,b_vocab)
          idx=torch.argmax(scores,1)
          captions[:,t]=idx
          prev_h=next_h

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
