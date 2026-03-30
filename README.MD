# AlphaZero-Quantum-Receiver

### AlphaZero-style Reinforcement Learning for KPSK State Discrimination

自进化量子接收机：基于 AlphaZero 风格强化学习的 KPSK 量子态判别

--- 3/26 Update ---

整理了代码，新增复杂kpsk环境，隔离不同功能，结构如下：

--scripts/

----run_benchmark.py （主程序入口，通过python scripts/run_benchmark.py执行）

--src/self_improving_quantum_receiver

----baselines （提供了helstrom和decision tree作为baselines, helstrom给出公式计算作为理论上限，decision tree利用贝叶斯后验在每一步选择最佳位移实现模拟上的最优解）

----benchmarks（主程序，训练RL，并对各种方案如helstrom/decision tree进行比较，给出准确率，计算资源等维度对比）

----environment（给出两种kpsk环境，一个是普通环境，K=4，2D平面上有四种候选状态，相邻态夹角为90°，一个是chirped环境，相位和每步幅度随着时间按类似正弦波动，模拟复杂的量子态情况）

----methods（AlphaZero like RL的主功能函数，给出RL框架，reward net + value net双网络结构，RL后辅助搜索以及模型训练过程）

## Overview | 项目简介

This project explores a learning-based approach for **quantum state discrimination** using an **AlphaZero-style reinforcement learning framework**.

We consider the problem of discriminating **K-ary phase-shift-keyed (KPSK) coherent states** using sequential adaptive measurements.

Instead of relying on handcrafted decision trees or analytic receiver designs (e.g., Dolinar-like receivers), our method learns measurement strategies through the interaction between:

- a **Policy Network**
- a **Value Network**
- a **Search-guided action refinement module**

The system progressively improves itself through **self-improvement training**, where search-refined actions are used to generate better training data for the neural networks.

---

本项目研究一种基于 **强化学习** 的量子态判别方法，采用 **AlphaZero 风格的自进化学习框架** 来设计自适应量子接收机。

研究对象为 **KPSK 相干态判别问题**。传统方法通常依赖解析接收机设计或基于规则的决策树（如 Dolinar-like receiver）。

本项目提出一种结合以下组件的学习框架：

- **策略网络（Policy Network）**
- **价值网络（Value Network）**
- **搜索引导动作优化（Search-guided refinement）**

通过搜索辅助生成训练数据，实现策略的 **self-improvement learning（自我改进学习）**。

---

# Problem Setup | 问题设置

We consider a communication system where the transmitter sends one of **K coherent states**

 \[
|\alpha_k\rangle = |\alpha e^{i2\pi k/K}\rangle
\] 

The receiver performs **sequential adaptive measurements**:

At each step:

1. Apply displacement \( \beta_t \)
2. Perform on/off photon detection
3. Update posterior probability using Bayes rule

After **T measurement steps**, the receiver outputs the most likely hypothesis.

Goal:

\[
\max P(\text{correct discrimination})
\]

---

我们研究的通信系统中，发送端发送 **K 个相干态之一**：

\[
|\alpha_k\rangle = |\alpha e^{i2\pi k/K}\rangle
\]

接收端执行 **序列自适应测量**：

每一步：

1. 施加位移 \( \beta_t \)
2. 进行 on/off 光子探测
3. 使用贝叶斯更新后验概率

在 **T 步测量后**：

\[
\hat{k} = \arg\max p_T(k)
\]

目标：

最大化判别正确率。

---

# Method | 方法

Our method follows an **AlphaZero-style learning framework**.

Instead of directly learning the optimal action, we combine:

### 1️⃣ Policy Network

Predicts a displacement proposal

$\[\beta_{prop} = \pi_\theta(s)\]$

Input state includes:

- posterior probability
- measurement step index
- remaining signal energy
- signal amplitude

---

### 2️⃣ Value Network

Estimates the probability of successful discrimination:

\[
V_\phi(s) \approx P(\text{correct} | s)
\]

This network allows fast evaluation of candidate actions.

---

### 3️⃣ Search-Guided Action Selection

At each step:

1. Policy proposes action \( \beta_{prop} \)
2. Generate nearby candidate displacements
3. Evaluate candidates using Value Network
4. Choose best displacement

\[
\beta_{exec} = \arg\max_{\beta} \text{score}(\beta)
\]

---

### 4️⃣ Self-Improvement Training

Training follows an iterative loop:

1. Run episodes using **policy + search**
2. Collect training data

\[
(s_t, \beta_{exec}, y)
\]

3. Train networks:

Policy learns refined actions

\[
\pi_\theta(s_t) \to \beta_{exec}
\]

Value learns final success probability

\[
V_\phi(s_t) \to y
\]

4. Improved networks generate better data in the next iteration.

---

# Evaluation | 模型评估

We evaluate the trained model using two modes.

### Policy-Only

Network directly outputs displacement:

\[
\beta_t = \pi_\theta(s_t)
\]

Measures how well the network approximates the search policy.

---

### Search-Guided

Network proposal is refined by local search:

\[
\beta_{prop} \rightarrow \beta_{exec}
\]

Measures the performance of the **complete system**.

---

Typical comparison:

| Method | Accuracy |
|------|------|
Decision Tree | baseline |
Policy Only | learned policy |
Policy + Search | AlphaZero-style system |

---
