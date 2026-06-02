# CS 224N 2026年冬季 作业 2
## Word2Vec 与 依存句法分析

**截止日期：1月22日，星期四，下午 4:30 PST**

在本作业中，你将复习 Word2Vec 背后的数学原理，并使用 PyTorch 构建一个神经依存句法分析器。有关 PyTorch 的基础知识复习，请查看 Canvas 上的 PyTorch 复习课。在第 1 部分中，你将探索使用朴素 softmax 损失训练 Word2vec 模型时涉及的偏导数。在第 2 部分中，你将学习两种通用的神经网络技术（Adam 优化和 Dropout）。在第 3 部分中，你将使用第 2 部分的技术实现并训练一个依存句法分析器，随后分析一些错误的依存句法解析案例。

如果你使用 LaTeX，可以使用 `\ifans{}` 命令来输入你的答案。
**请在 Gradescope 上正确标记问题，否则助教（TA）会因未标记问题而扣分。**

---

## 1. 理解 word2vec (20 分)

回想一下，word2vec 的核心见解是“**一个词的含义可以通过它的邻居来了解**”（*a word is known by the company it keeps*）。具体来说，考虑一个由特定长度的上下文包围的“中心”词 $c$。我们将此上下文窗口中的词称为“外部词”（$O$）。例如，在图 1 中，上下文窗口长度为 2，中心词 $c$ 是 "banking"，外部词是 "turning"、"into"、"crises" 和 "as"：

**(此处为图 1：窗口大小为 2 的 word2vec skip-gram 预测模型示例)**

Skip-gram word2vec 旨在学习概率分布 $P(O|C)$。具体而言，给定一个特定的词 $o$ 和一个特定的词 $c$，我们想要预测 $P(O = o | C = c)$：即词 $o$ 是词 $c$ 的“外部”词（即它落在 $c$ 的上下文窗口内）的概率。我们通过对一系列向量点积取 **softmax** 函数来模拟这一概率：

$$P(O = o | C = c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in Vocab} \exp(u_w^T v_c)} \quad (1)$$

对于每个词，我们学习向量 $u$ 和 $v$，其中 $u_o$ 是代表外部词 $o$ 的“外部”向量，$v_c$ 是代表中心词 $c$ 的“中心”向量。我们将这些参数存储在两个矩阵 $\mathbf{U}$ 和 $\mathbf{V}$ 中。$\mathbf{U}$ 的各列是所有的外部向量 $u_w$；$\mathbf{V}$ 的各列是所有的中心向量 $v_w$。$\mathbf{U}$ 和 $\mathbf{V}$ 都包含词汇表中每个词 $w$ 的向量。

---

**第 2 页内容：**

回想课程内容，对于单个词对 $c$ 和 $o$，损失函数由下式给出：

$$J_{naive-softmax}(v_c, o, \mathbf{U}) = -\log P(O = o | C = c) \quad (2)$$

我们可以将此损失视为真实分布 $y$ 与预测分布 $\hat{y}$ 之间的**交叉熵**。真实经验分布 $y$ 是一个独热（one-hot）向量，在真实外部词 $o$ 处为 1，其余位置为 0。预测分布 $\hat{y}$ 是由等式 (1) 给出的概率分布 $P(O|C=c)$。

**注意：** 在计算导数时，请使用课堂上介绍的方法（即不要使用泰勒级数近似）。

(a) (2 分) 证明朴素 softmax 损失（等式 2）与 $y$ 和 $\hat{y}$ 之间的交叉熵损失相同：
$$-\sum_{w \in Vocab} y_w \log(\hat{y}_w) = -\log(\hat{y}_o) \quad (3)$$

$$ -log(\hat{y_o})=-logP(O=o|C=c)=J $$

(b) (6 分) i. 计算 $J_{naive-softmax}$ 对 $v_c$ 的偏导数。请用 $y, \hat{y}$ 和 $\mathbf{U}$ 来表示。
*   **注意：** 必须遵循 **形状约定（shape convention）**：偏导数的形状应与变量形状一致。
*   请提供向量化的答案。

$$J=-log(\hat{y_o})=-u_o^Tv_c+log\sum_w{e^{u_w^T v_c}} $$

左部分的偏导为 $-u_o$ 右部分的偏导为 $\frac{\sum_x{e^{u_x^T v_c}}}{\sum_w{e^{u_w^T v_c}}}=\sum_w{\hat{y_w}u_w}=U\hat{y}$

$$\frac{\partial J}{\partial v_c}=U\hat{y}-u_o=U(\hat{y}-y)$$

ii. 你计算的梯度何时等于零？写出一个数学等式。

$$\hat{y}=y$$

iii. 提供对该梯度的解释：当从 $v_c$ 中减去该梯度时，这两个项分别如何改进词向量？

$$-\frac{\partial J}{\partial v_c}=u_o-U\hat{y}$$

使得 $v_c$ 向着 $u_o$ 靠近，向着其他向量远离

(c) (1 分) 在下游应用中，通常使用 **L2 归一化向量** ($u/\|u\|_2$)。考虑一个正负短语分类任务，通过词向量求和后的符号来判断。L2 归一化何时会消除有用信息？何时不会？

取决于任务是否在乎语义的强度

---

**第 4 页内容：**

## 2. 神经网络优化 (8 分)

**(a) Adam 优化器 (4 分)**
i. (2 分) Adam 使用 **动量（momentum）** 技巧，通过保留梯度的滚动平均值 $m$ 来工作。简要解释（2-4句话）使用 $m$ 如何防止更新波动过大，以及为什么这种低方差对学习有帮助。

$m$ 的值大部分来自于先前的梯度，使得梯度的更新更加平滑

可以防止极端数据带来的波动

ii. (2 分) Adam 通过 **自适应学习率** 扩展了动量的思想，记录了梯度幅度的滚动平均值 $v$。Adam 会给哪些参数更大的更新？为什么这有助于学习？

会给历史模长较小的参数更大的更新，使得模型在所有维度上的学习更加均匀

**(b) Dropout (4 分)**
Dropout 是一种正则化技术。在训练期间，随机以概率 $p_{drop}$ 将隐藏层 $h$ 中的单元设置为 0，然后乘以常数 $\gamma$。
$$h_{drop} = \gamma d \odot h$$
i. (2 分) 为了使 $h_{drop}$ 的期望值等于 $h$，$\gamma$ 应该等于什么（用 $p_{drop}$ 表示）？

$$\frac{1}{1-p_{drop}}$$

ii. (2 分) 为什么 Dropout 应该在训练时应用，而在评估时不应用？

在训练时可以防止过拟合以获得更加优秀的模型，评估时因充分利用训练好的模型

---

**第 5-6 页内容：**

## 3. 基于神经状态转换的依存句法分析 (40 分)

在本节中，你将实现一个基于神经网络的依存句法分析器，目标是最大化 **UAS（未标注附件得分）**。

依存句法分析器分析句子的语法结构，确定“核心词”（head words）和修饰这些核心词的词之间的关系。你将实现一个**基于状态转换的解析器**，它通过以下方式维护解析状态：
*   一个存储当前处理单词的 **stack（栈）**。
*   一个存储待处理单词的 **buffer（缓存）**。
*   一个解析器预测的 **dependencies（依存关系列表）**。

**转换动作：**
*   **SHIFT**: 从 buffer 移除第一个词并压入 stack。
*   **LEFT-ARC**: 将 stack 中倒数第二个词标记为倒数第一个词的从属词，并将其从 stack 移除。
*   **RIGHT-ARC**: 将 stack 中最后一个词标记为倒数第二个词的从属词，并将其从 stack 移除。

**(a) (4 分)** 给出句子 "I presented my findings at the NLP conference" 的转换序列追踪表。

presented->I

findings->my

presented->findings

conference->NLP

conference->the

conference->at

presented->conference

presented->root


**(b) (2 分)** 一个包含 $n$ 个词的句子经过多少步（用 $n$ 表示）完成解析？简要说明原因。

2n,需要先入栈再出栈

**(d) (8 分)** 实现 `parser_transitions.py` 中的 `minibatch_parse` 算法。该算法允许模型同时处理一批句子的解析转换，以提高效率。

**(e) (20 分) 训练神经网络：**
你将构建一个模型，输入 stack 和 buffer 的状态特征，预测下一步应该执行的操作（SHIFT, L-ARC, R-ARC）。
*   **特征提取：** 提取当前 stack 顶和 buffer 前端的单词及其词向量。
*   **网络结构：**
    $$h = ReLU(xW + b_1)$$
    $$l = hU + b_2$$
    $$\hat{y} = softmax(l)$$
*   **注意：** 禁止使用 `torch.nn.Linear` 或 `torch.nn.Embedding`，必须手动定义权重参数。

---

**第 8 页：提交说明**
你需要在 Gradescope 上提交两个部分：
1.  **代码部分**：运行 `collect_submission.sh` 生成 `assignment2.zip`。
2.  **书面部分**：提交 PDF 格式的数学推导和分析。

**目标指标：**
在非 debug 模式下，你的模型在开发集上的 UAS 应达到 **87** 以上。训练过程在完整数据集上不应超过 15 分钟。