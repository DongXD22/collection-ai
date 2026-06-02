### 理解 word2vec (20 分)

回想一下，`word2vec` 背后的核心见解是*“一个词的含义可以通过它的邻居来了解”*（a word is known by the company it keeps）。具体来说，考虑一个“中心”（center）词 $c$，其前后被一定长度的上下文所包围。我们将这个上下文窗口中的词称为“外部词”（outside words，记为 $O$）。例如，在图 1 中，上下文窗口长度为 2，中心词 $c$ 是 `banking`，而外部词是 `turning`、`into`、`crises` 和 `as`：

Skip-gram `word2vec` 旨在学习概率分布 $P(O|C)$。具体而言，给定一个特定的词 $o$ 和一个特定的词 $c$，我们想要预测 $P(O=o|C=c)$：即词 $o$ 是 $c$ 的“外部”词（即它落在 $c$ 的上下文窗口内）的概率。我们通过对一系列向量点积取 softmax 函数来对这个概率进行建模：

$$ P(O=o \mid C=c) = \frac{\exp(\mathbf{u}_{o}^\top \mathbf{v}_c)}{\sum_{w \in \text{Vocab}} \exp(\mathbf{u}_{w}^\top \mathbf{v}_c)} \quad (1) $$

对于每个词，我们学习向量 $u$ 和 $v$，其中 $\mathbf{u}_o$ 是代表外部词 $o$ 的“外部”向量，$\mathbf{v}_c$ 是代表中心词 $c$ 的“中心”向量。我们将这些参数存储在两个矩阵 $\mathbf{U}$ 和 $\mathbf{V}$ 中。$\mathbf{U}$ 的列是所有的“外部”向量 $\mathbf{u}_w$；$\mathbf{V}$ 的列是所有的“中心”向量 $\mathbf{v}_w$。$\mathbf{U}$ 和 $\mathbf{V}$ 都包含词汇表中每个 $w \in \text{Vocabulary}$ 的向量。[^1]

回顾课程内容，对于单个词对 $c$ 和 $o$，损失由下式给出：

$$ \mathbf{J}_{\text{naive-softmax}}(\mathbf{v}_c, o, \mathbf{U}) = -\log P(O=o \mid C=c) \quad (2) $$

我们可以将此损失视为对于特定的中心词 $c$ 和外部词 $o$，真实分布 $\mathbf{y}$ 和预测分布 $\hat{\mathbf{y}}$ 之间的交叉熵（cross-entropy）[^2]。在这里，$\mathbf{y}$ 和 $\hat{\mathbf{y}}$ 都是长度等于词汇表中词数的向量。此外，这些向量中的第 $k$ 个元素表示在给定 $c$ 的情况下，第 $k$ 个词是“外部词”的条件概率。对于中心词 $c$ 和外部词 $o$ 的这个特定示例，真实的经验分布 $\mathbf{y}$ 是一个独热（one-hot）向量，其中真实外部词 $o$ 对应的值为 1，其他所有位置的值为 0 [^3]。预测分布 $\hat{\mathbf{y}}$ 是由我们的模型在公式 (1) 中给出的概率分布 $P(O \mid C=c)$。

**注意：** 在整个作业中，计算导数时，请使用课堂上复习的方法（即不要使用泰勒级数近似）。

**(a) (2 分)**
证明朴素 softmax 损失（公式 2）与 $\mathbf{y}$ 和 $\hat{\mathbf{y}}$ 之间的交叉熵损失相同，即（注意 $\mathbf{y}$（真实分布）、$\hat{\mathbf{y}}$（预测分布）是向量，而 $\hat{\mathbf{y}}_o$ 是标量）：

$$ -\sum_{w \in \text{Vocab}} \mathbf{y}_w \log(\hat{\mathbf{y}}_w) = - \log (\hat{\mathbf{y}}_o) \quad (3) $$

$$ -log(\hat{y_o})=-logP(O=o|C=c)=J $$

你的答案应该只有一行。你可以用文字描述你的答案。

**(b) (6 分)**
1. 计算 $\mathbf{J}_{\text{naive-softmax}}(\mathbf{v}_c, o, \mathbf{U})$ 关于 $\mathbf{v}_c$ 的偏导数。*请用 $\mathbf{y}$、$\hat{\mathbf{y}}$ 和 $\mathbf{U}$ 来表示你的答案，并展示你的计算过程以获得满分*。
   * **注意**：你最终的偏导数答案应遵循形状约定（shape convention）：任何函数 $f(x)$ 关于 $x$ 的偏导数应与 $x$ 具有**相同的形状**[^4]。
   * 请以向量化形式提供偏导数的答案。例如，当我们要求你用 $\mathbf{y}$、$\hat{\mathbf{y}}$ 和 $\mathbf{U}$ 来写答案时，你不能在最终答案中引用这些项的具体元素（例如 $\mathbf{y}_1, \mathbf{y}_2, \dots$）。你也不能引用具体的向量，例如 $u_0, u_1$ 等。

    
$$J=-log(\hat{y_o})=-u_o^Tv_c+log\sum_w{e^{u_w^T v_c}} $$

左部分的偏导为 $-u_o$ 右部分的偏导为 $\frac{\sum_x{e^{u_x^T v_c}}}{\sum_w{e^{u_w^T v_c}}}=\sum_w{\hat{y_w}u_w}=U\hat{y}$

$$\frac{\partial J}{\partial v_c}=U\hat{y}-u_o=U(\hat{y}-y)$$

2. 你计算的梯度何时等于零？写出一个数学等式。
   **提示：** 你可能需要复习并使用一些基础的线性代数概念。

   $$\hat{y}=y$$

3. 你求出的梯度是两项之差。请解释：当从词向量 $\mathbf{v}_c$ 中减去这个梯度时，这两项分别是如何改进词向量的。

$$-\frac{\partial J}{\partial v_c}=u_o-U\hat{y}$$

使得 $v_c$ 向着 $u_o$ 靠近，向着其他向量远离


**(c) (1 分)**
在许多使用词嵌入的下游应用中，通常会使用 L2 归一化向量（例如 $\mathbf{u}/||\mathbf{u}||_2$，其中 $||\mathbf{u}||_2 = \sqrt{\sum_i u_i^2}$）而不是它们的原始形式（例如 $\mathbf{u}$）。让我们考虑一个假设的下游任务：将短语进行二元分类，分为正面或负面，你基于单词独立嵌入的总和来决定正负号。在什么时候 L2 归一化会消除下游任务的有用信息？什么时候不会？

**提示：** 考虑这样一种情况，对于某些词 $x \neq y$ 和某个标量 $\alpha$，存在 $\mathbf{u}_x = \alpha\mathbf{u}_y$。当 $\alpha$ 为正时，归一化后的 $\mathbf{u}_x$ 和归一化后的 $\mathbf{u}_y$ 的值分别是多少？$\mathbf{u}_x$ 和 $\mathbf{u}_y$ 之间应该存在怎样的关系，才能使这种归一化影响或不影响最终的分类结果？

取决于任务是否在乎语义的强度

**(d) (1 分)**
写下 $\mathbf{J}_{\text{naive-softmax}}(\mathbf{v}_c, o, \mathbf{U})$ 关于 $\mathbf{U}$ 的偏导数。请将你的答案分解为列向量的形式 $\frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_1}, \frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_2}, \cdots, \frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_{|\text{Vocab}|}}$（不要进一步展开这些项）。不需要推导过程，只需以矩阵的形式给出答案。

$$
\frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{U}} = \begin{bmatrix} \frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_1}, \frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_2}, \cdots, \frac{\partial \mathbf{J}(\mathbf{v}_c, o, \mathbf{U})}{\partial \mathbf{u}_{|Vocab|}} \end{bmatrix}
$$

**(e) (5 分)**
计算 $\mathbf{J}_{\text{naive-softmax}}(\mathbf{v}_c, o, \mathbf{U})$ 关于每个“外部”词向量 $\mathbf{u}_w$ 的偏导数。这里有两种情况：当 $w=o$（即真实的“外部”词向量）时，以及 $w \neq o$（针对所有其他词）时。请用 $\mathbf{y}$、$\hat{\mathbf{y}}$ 和 $\mathbf{v}_c$ 来表示你的答案。在这个小题中，你也可以使用这些项中的具体元素（例如 $\mathbf{y}_1, \mathbf{y}_2, \dots$）。注意 $\mathbf{u}_w$ 是一个向量，而 $\mathbf{y}_1, \mathbf{y}_2, \dots$ 是标量。展示你的计算过程以获得满分。

$$\frac{\partial J}{\partial U}=V_c(\hat{y}-y)^T$$

**(f) (2 分)**
作为本题的额外练习，你将对一些常见的损失函数求导，这些函数可能会用于 `word2vec` 的变体（比如负采样变体）。Leaky ReLU（带泄漏的线性整流单元）激活函数由公式 4 和图 2 给出：

$$ f(x) = \max(\alpha x, x) \quad (4) $$


$$\alpha,1$$

其中 $x$ 是一个标量且 $0<\alpha <1$，请计算 $f(x)$ 关于 $x$ 的导数。你可以忽略在 0 处导数未定义的情况。[^5]

**(g) (3 分)**
Sigmoid 函数由公式 5 给出：

$$ \sigma (x) = \frac{1}{1 + e^{-x}} = \frac{e^{x}}{e^{x} + 1} \quad (5) $$

请计算 $\sigma(x)$ 关于 $x$ 的导数，其中 $x$ 是一个标量。请用 $\sigma(x)$ 来表示你的答案。展示你的计算过程以获得满分。

$$\frac{\sigma(x)}{e^x+1}$$
---

### 脚注补充说明

[^1]: 假设词汇表中的每个词都与一个整数 $k$ 对应。加粗的小写字母代表向量。$\mathbf{u}_k$ 既是矩阵 $\mathbf{U}$ 的第 $k$ 列，也是索引为 $k$ 的词的“外部”词向量。$\mathbf{v}_k$ 既是矩阵 $\mathbf{V}$ 的第 $k$ 列，也是索引为 $k$ 的词的“中心”词向量。**为了简化符号，我们将交替使用 $k$ 来指代词 $k$ 以及词 $k$ 的索引。**
[^2]: 真实（离散）概率分布 $p$ 和另一个分布 $q$ 之间的**交叉熵损失**为 $-\sum_i p_i \log(q_i)$。
[^3]: 请注意，整个训练数据集中上下文词的真实条件概率分布不会是独热（one-hot）的。
[^4]: 这使得我们能够使用梯度下降来高效地最小化函数，而不必担心重塑（reshaping）或维度不匹配的问题。在遵循形状约定的同时，我们能保证 $\theta := \theta - \alpha\frac{\partial J(\theta)}{\partial \theta}$ 是一个定义良好的更新规则。
[^5]: 如果你对此时如何处理导数感兴趣，可以阅读更多关于[次导数（subderivatives）](https://en.wikipedia.org/wiki/Subderivative)的概念。

### 2. 神经网络优化 (8 分)

**(a) (4 分) Adam 优化器**

回想一下标准的随机梯度下降（Stochastic Gradient Descent）更新规则：
$$ \theta_{t+1} \leftarrow \theta_t - \alpha \nabla_{\theta_t} J_{\text{minibatch}}(\theta_t) $$
其中 $t + 1$ 是当前的时间步，$\theta$ 是一个包含所有模型参数的向量（$\theta_t$ 是在时间步 $t$ 的模型参数，$\theta_{t+1}$ 是在时间步 $t + 1$ 的模型参数），$J$ 是损失函数，$\nabla_{\theta_t} J_{\text{minibatch}}(\theta)$ 是损失函数对参数在数据小批量（minibatch）上的梯度，$\alpha$ 是学习率。Adam 优化器[^6] 使用了更复杂的更新规则，包含两个额外的步骤。[^7]

i. (2 分) 首先，Adam 使用了一种称为**动量（momentum）**的技巧，通过跟踪梯度的滚动平均值 $\mathbf{m}$ 来实现：
$$ \mathbf{m}_{t+1} \leftarrow \beta_1 \mathbf{m}_t + (1 - \beta_1)\nabla_{\theta_t} J_{\text{minibatch}}(\theta_t) $$
$$ \theta_{t+1} \leftarrow \theta_t - \alpha \mathbf{m}_{t+1} $$
其中 $\beta_1$ 是一个介于 0 和 1 之间的超参数（通常设置为 0.9）。**请用 2-4 句话简要解释**（你不需要进行数学证明，只需给出直观的理解），使用 $\mathbf{m}$ 如何防止更新变化过大，以及为什么这种低方差总体上可能对学习有帮助。

$m$ 的值大部分来自于先前的梯度，使得梯度的更新更加平滑

可以防止极端数据带来的波动

ii. (2 分) Adam 进一步扩展了动量的思想，引入了**自适应学习率（adaptive learning rates）**的技巧，通过跟踪梯度幅度的滚动平均值 $\mathbf{v}$ 来实现：
$$ \mathbf{m}_{t+1} \leftarrow \beta_1 \mathbf{m}_t + (1 - \beta_1)\nabla_{\theta_t} J_{\text{minibatch}}(\theta_t) $$
$$ \mathbf{v}_{t+1} \leftarrow \beta_2 \mathbf{v}_t + (1 - \beta_2)(\nabla_{\theta_t} J_{\text{minibatch}}(\theta_t) \odot \nabla_{\theta_t} J_{\text{minibatch}}(\theta_t)) $$
$$ \theta_{t+1} \leftarrow \theta_t - \alpha \mathbf{m}_{t+1} / \sqrt{\mathbf{v}_{t+1}} $$
其中 $\odot$ 和 $/$ 表示逐元素的乘法和除法（所以 $\mathbf{z} \odot \mathbf{z}$ 就是逐元素求平方），$\beta_2$ 是一个介于 0 和 1 之间的超参数（通常设置为 0.99）。由于 Adam 将更新量除以 $\sqrt{\mathbf{v}}$，模型中的哪些参数将获得更大的更新？为什么这可能有助于学习？**请用 2-4 句话简要解释**。

会给历史模长较小的参数更大的更新，使得模型在所有维度上的学习更加均匀

**(b) (4 分) Dropout**[^8] 是一种正则化技术。在训练期间，Dropout 随机地以概率 $p_{\text{drop}}$ 将隐藏层 $\mathbf{h}$ 中的单元设为零（在每个 minibatch 中丢弃不同的单元），然后将 $\mathbf{h}$ 乘以一个常数 $\gamma$。我们可以将其写为：
$$ \mathbf{h}_{\text{drop}} = \gamma \mathbf{d} \odot \mathbf{h} $$
其中 $\mathbf{d} \in \{0, 1\}^{D_h}$（$D_h$ 是 $\mathbf{h}$ 的大小）是一个掩码向量，其中每个元素以概率 $p_{\text{drop}}$ 为 0，以概率 $(1 - p_{\text{drop}})$ 为 1。选择 $\gamma$ 是为了确保 $\mathbf{h}_{\text{drop}}$ 的期望值等于 $\mathbf{h}$：
$$ \mathbb{E}_{p_{\text{drop}}} [\mathbf{h}_{\text{drop}}]_i = h_i $$
对于所有的 $i \in \{1, \dots, D_h\}$。

i. (2 分) 用 $p_{\text{drop}}$ 表示，$\gamma$ 必须等于什么？简要说明你的答案或使用上面给出的公式展示你的数学推导。

$$\frac{1}{1-p_{drop}}$$

ii. (2 分) 为什么应该在训练期间应用 Dropout？为什么在评估（evaluation）时不应该应用 Dropout？**请用 2-4 句话简要解释。** **提示：** 查看链接中的 Dropout 论文可能会对你有所帮助。

在训练时可以防止过拟合以获得更加优秀的模型，评估时因充分利用训练好的模型

---

### 3. 基于神经状态转换的依存句法分析 (40 分)

在本节中，你将实现一个基于神经网络的依存句法分析器（dependency parser），目标是在 **UAS（Unlabeled Attachment Score，未标注依存得分）** 指标上实现性能最大化。

在开始之前，请按照 README 文件的指示安装作业所需的所有依赖项。我们将使用来自 `https://pytorch.org/get-started/locally/` 的 PyTorch 2.1.2 版本，并禁用 CUDA 选项（设置为 None），以及使用 `tqdm` 包——它能在你的训练过程中生成进度条可视化。官方的 PyTorch 网站是一个极好的资源，包含了用于理解 PyTorch Tensor 库和神经网络的教程。

依存句法分析器分析句子的语法结构，建立*核心词（head words）*以及修饰这些核心词的词之间的关系。依存句法分析器有多种类型，包括基于状态转换的分析器（transition-based parsers）、基于图的分析器（graph-based parsers）和基于特征的分析器（feature-based parsers）。你的实现将是一个**基于状态转换的分析器**，它每次一步地增量构建解析树。在每一步中，它都维护一个*部分解析（partial parse）*，表示如下：
*   一个正在处理的单词的 **栈（stack）**。
*   一个待处理单词的 **缓存（buffer）**。
*   一个由分析器预测的 **依存关系（dependencies）** 列表。

最初，栈仅包含 `ROOT`，依存关系列表为空，缓存包含句子中的所有单词（按顺序排列）。在每一步中，解析器对部分解析应用一次**转换（transition）**，直到其缓存为空且栈的大小为 1。可以应用以下转换操作：

*   **SHIFT**: 从缓存中移除第一个单词并将其压入栈。
*   **LEFT-ARC**: 将栈上倒数第二个（第二新添加的）项目标记为栈顶第一项的从属词，并将第二个项目从栈中移除，向依存关系列表中添加一个 `first_word → second_word` 的依存关系。
*   **RIGHT-ARC**: 将栈上第一个（最新添加的）项目标记为倒数第二项的从属词，并将第一个项目从栈中移除，向依存关系列表中添加一个 `second_word → first_word` 的依存关系。

在每一步，你的分析器将使用神经网络分类器在三种转换操作中做出决定。

**(a) (4 分)** 走一遍解析句子 “*I presented my findings at the NLP conference*” 所需的转换序列。该句子的依存关系树如下所示。在每一步，给出栈和缓存的配置状态，以及本步应用的转换操作和添加的新依存关系（如果有）。前三个步骤如下表所示作为示例。

presented->I

findings->my

presented->findings

conference->NLP

conference->the

conference->at

presented->conference

presented->root

*(此处应有一张依存关系树的图)*

| Stack (栈) | Buffer (缓存) | New dependency (新依存关系) | Transition (转换操作) |
| :--- | :--- | :--- | :--- |
| `[ROOT]` | `[I, presented, my, findings, at, the, NLP, conference]` | | Initial Configuration |
| `[ROOT, I]` | `[presented, my, findings, at, the, NLP, conference]` | | `SHIFT` |
| `[ROOT, I, presented]` | `[my, findings, at, the, NLP, conference]` | | `SHIFT` |
| `[ROOT, presented]` | `[my, findings, at, the, NLP, conference]` | `presented→I` | `LEFT-ARC` |

**(b) (2 分)** 一个包含 $n$ 个单词的句子将被解析多少步（用 $n$ 表示）？请用 1-2 句话简要解释原因。

2n,需要先入栈再出栈

**(c) (6 分)** 在 `parser_transitions.py` 中的 `PartialParse` 类里实现 `__init__` 和 `parse_step` 函数。这实现了你的分析器将使用的状态转换机制。你可以通过运行 `python parser_transitions.py part_c` 来运行基本（非详尽）的测试。

**(d) (8 分)** 我们的网络将预测下一个应该应用于部分解析的转换操作。我们可以使用它来解析单个句子，方法是一直应用预测的转换操作，直到解析完成。然而，当神经网络同时对**批量（batches）**数据进行预测时，其运行效率会高得多（即，同时预测任何不同的部分解析的下一次转换）。我们可以使用以下算法来解析小批量（minibatches）句子。

---
**算法 1：小批量依存句法分析 (Minibatch Dependency Parsing)**
**输入:** `sentences`，一个待解析的句子列表；`model`，我们用来做出解析决定的模型。

将 `partial_parses` 初始化为 `PartialParses` 列表，为 `sentences` 中的每个句子创建一个。

将 `unfinished_parses` 初始化为 `partial_parses` 的浅拷贝（shallow copy）。

**while** `unfinished_parses` 不为空 **do**

&nbsp;&nbsp;&nbsp;&nbsp;从 `unfinished_parses` 中取出前 `batch_size` 个解析作为一个小批量（minibatch）。

&nbsp;&nbsp;&nbsp;&nbsp;使用 `model` 预测该小批量中每个部分解析的下一步转换操作。

&nbsp;&nbsp;&nbsp;&nbsp;使用预测的转换操作，对小批量中的每个部分解析执行一次解析步骤 (`parse_step`)。

&nbsp;&nbsp;&nbsp;&nbsp;从 `unfinished_parses` 中移除已完成的解析（缓存为空且栈大小为 1）。

**end while**

**返回:** `partial_parses` 中每个（现在已完成的）解析的依存关系。
---

在 `parser_transitions.py` 中的 `minibatch_parse` 函数里实现此算法。你可以通过运行 `python parser_transitions.py part_d` 来运行基本（非详尽）的测试。

*注：你需要正确实现 `minibatch_parse` 来评估你在第 (e) 部分构建的模型。但是，训练该模型不需要它，所以即使你还没有实现 `minibatch_parse`，你也应该能完成 (e) 部分的大部分内容。*

**(e) (20 分)** 现在我们将训练一个神经网络，给定栈、缓存和依存关系的状态，预测应该应用哪种转换操作。
首先，模型提取一个代表当前状态的特征向量。我们将使用原始神经依存分析论文《A Fast and Accurate Dependency Parser using Neural Networks》[^9] 中提出的特征集。提取这些特征的函数已经为你实现在 `utils/parser_utils.py` 中。这个特征向量由一个**词元（tokens）列表**组成（例如，栈中的最后一个词、缓存中的第一个词、栈中倒数第二个词的从属词（如果有的话）等）。它们可以表示为一个整数列表 $\mathbf{w} =[w_1, w_2, \dots, w_m]$，其中 $m$ 是特征的数量，并且每个 $0 \leq w_i < |V|$ 是词汇表（$|V|$ 是词汇表大小）中某个词元的索引。然后，我们的网络查找每个词的嵌入，并将它们拼接（concatenates）成一个单一的输入向量：

$$ \mathbf{x} = [\mathbf{E}_{w_1}, \dots, \mathbf{E}_{w_m}] \in \mathbb{R}^{dm} $$

其中 $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ 是一个嵌入矩阵（embedding matrix），它的每一行 $\mathbf{E}_w$ 是特定词 $w$ 维度为 $d$ 的向量。然后我们进行如下计算来进行预测：

$$ \mathbf{h} = \text{ReLU}(\mathbf{x}\mathbf{W} + \mathbf{b}_1) $$
$$ \mathbf{l} = \mathbf{h}\mathbf{U} + \mathbf{b}_2 $$
$$ \hat{\mathbf{y}} = \text{softmax}(\mathbf{l}) $$

其中 $\mathbf{h}$ 被称为隐藏层（hidden layer），$\mathbf{l}$ 被称为对数几率（logits），$\hat{\mathbf{y}}$ 被称为预测值（predictions），且 $\text{ReLU}(z) = \max(z, 0)$。我们将训练模型以最小化交叉熵损失：

$$ J(\theta) = CE(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^3 y_j \log \hat{y}_j $$

其中 $y_j$ 表示 $y$ 的第 $j$ 个元素。为了计算训练集的损失，我们将所有训练样本的 $J(\theta)$ 求平均。

i. 计算 $\mathbf{h} = \text{ReLU}(\mathbf{x}\mathbf{W} + \mathbf{b}_1)$ 关于 $\mathbf{x}$ 的导数。为简单起见，你只需要展示对于某个索引 $i$ 和 $j$ 的偏导数 $\frac{\partial h_i}{\partial x_j}$。你可以忽略在 0 处导数未定义的情况。

$$\mathbf{W}^T_{ji},0$$

ii. 回顾在第 1b 部分，我们计算了 $J_{\text{naive-softmax}}(\mathbf{v}_c, o, \mathbf{U})$ 的偏导数。同样地，请计算 $J(\theta)$ 关于 $\mathbf{l}$ 的第 $i$ 个条目（记为 $l_i$）的偏导数。具体来说，计算 $\frac{\partial CE(\mathbf{y}, \hat{\mathbf{y}})}{\partial l_i}$，假设 $\mathbf{l} \in \mathbb{R}^3$, $\hat{\mathbf{y}} \in \mathbb{R}^3$, $\mathbf{y} \in \mathbb{R}^3$，并且真实的标签是 $c$（即，如果 $j=c$ 则 $y_j = 1$）。**提示：**使用链式法则：$\frac{\partial J}{\partial l_i} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial l_i}$。

$$\hat{y_i}-y_i$$

iii. 我们将使用 **UAS (Unlabeled Attachment Score)** 分数作为评估指标。UAS 是指预测正确的依存关系数量与总依存关系数量（无论关系标签类型如何，因为我们的模型不预测标签）的比率。

在 `parser_model.py` 中，你会找到使用 PyTorch 实现这个简单神经网络的框架代码。完成 `__init__`、`embedding_lookup` 和 `forward` 函数来实现该模型。然后完成 `run.py` 文件中的 `train_for_epoch` 和 `train` 函数。
最后，执行 `python run.py` 来训练你的模型，并在 Penn Treebank（带有通用依存关系标注）的测试数据上计算预测结果。

**注意：**
*   在此作业中，你需要手动实现线性层（Linear layer）和嵌入层（Embedding layer）。请**绝对不要**在你的代码中使用 `torch.nn.Linear` 或 `torch.nn.Embedding` 模块，否则此题将被扣分。
*   如果 TODO 中有明确的变量名要求，请务必遵循我们的命名规范，以便获得满分。如果没有明确要求，你可以自由声明其他变量名。

**提示：**
*   要求你声明的变量（`self.embed_to_hidden_weight`, `self.embed_to_hidden_bias`, `self.hidden_to_logits_weight`, `self.hidden_to_logits_bias`）分别对应于上述公式中的变量 ($\mathbf{W}, \mathbf{b}_1, \mathbf{U}, \mathbf{b}_2$)。
*   在算法中反向推导（从 $\hat{\mathbf{y}}$ 开始）并随时跟踪矩阵/向量的尺寸可能会有所帮助。
*   一旦你实现了 `embedding_lookup` (-e) 或 `forward` (-f) 函数，你可以通过运行带有 `-e` 或 `-f` 或同时带有两者的标志 `python parser_model.py` 来运行针对每个函数的基本完整性检查（sanity checks）。这些检查非常基础，通过它们并不意味着你的代码就没有 bug。
*   在调试（debugging）时，你可以添加一个调试标志：`python run.py -d`。这会让代码在一小部分数据子集上运行，从而不需要太长的训练时间。一旦你完成调试，请确保移除 `-d` 标志以在完整数据上运行模型。
*   在启用 debug 模式运行时，你应该能在开发集（dev set）上获得低于 0.2 的损失和高于 65 的 UAS（尽管在极少数情况下结果可能更低，因为训练存在随机性）。
*   在禁用 debug 模式下，整个训练数据集的训练时间不应超过 **15分钟**。
*   在禁用 debug 模式下，你应该能在训练集上获得低于 0.08 的损失，并在开发集上获得高于 87 的 UAS 分数。作为对比，原始神经依存分析论文中的模型达到了 92.5 的 UAS。如果你愿意，你可以微调模型的超参数（隐藏层大小、Adam 的超参数、epoch 数量等）来提升性能（但这不是硬性要求）。

**交付成果 (Deliverables):**
*   在 `parser_transitions.py` 中神经依存分析器使用的状态转换机制的有效实现。
*   在 `parser_transitions.py` 中小批量依存分析的有效实现。
*   在 `parser_model.py` 中神经依存分析器的有效实现。（我们将会查看并运行此代码以进行评分）。
*   在 `run.py` 中训练函数的有效实现。（我们将会查看并运行此代码以进行评分）。
*   在实现 `embedding_lookup` 时，请使用高效的函数并**避免使用 for 循环**。否则，你可能会超出 GradeScope 的测试时间限制。
*   在你的**书面提交报告**中，报告你的模型在开发集（dev set）上达到的**最佳 UAS**，以及在测试集（test set）上达到的 **UAS**。你可以在 PDF 中报告此项内容并标记该页面。

---

### 提交说明 (Submission Instructions)
你应该在 GradeScope 上将此作业作为两次独立的提交进行提交 —— 一次用于 Assignment 2 [coding]（代码），另一次用于 Assignment 2[written]（书面）：
1.  运行 `collect_submission.sh` 脚本以生成你的 `assignment2.zip` 文件。
2.  将你的 `assignment2.zip` 文件上传到 GradeScope 上的 Assignment 2 [coding]。
3.  将你的书面解答上传到 GradeScope 上的 Assignment 2 [written]。

---

### 脚注补充说明

[^6]: Kingma and Ba, 2015, https://arxiv.org/pdf/1412.6980.pdf
[^7]: 实际的 Adam 更新使用了几个额外但不太重要的技巧，但我们在这里不用担心它们。如果你想了解更多，可以查看：http://cs231n.github.io/neural-networks-3/#sgd
[^8]: Srivastava et al., 2014, https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf[^9]: Chen and Manning, 2014, https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf