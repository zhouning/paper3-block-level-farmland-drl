# Paper 3: Block-Level MDP 模型定义

## 1. 问题背景

在全域土地综合整治中，决策者需要在有限的整治预算下，选择在哪些区域投入资源进行耕地-林地置换（林耕置换），以实现：
1. **降低耕地平均坡度**（耕地上山→下山）
2. **提高耕地空间连通性**（形成百亩方/千亩方连片耕地）

Paper 2 证明了地块级（parcel-level）MDP 在大规模真实数据上失效——近独立性（near-independence）使 DRL 无法发挥序贯决策优势。Paper 3 将决策尺度从地块提升到地块组（block），恢复了有意义的序贯依赖关系。

---

## 2. 空间抽象：地块→地块组

### 2.1 地块组定义方法

采用混合方法（Hybrid Barrier Segmentation + Agglomerative Clustering）：

1. **提取可置换地块**：筛选耕地（DLBM 011/012/013）和林地（031/032/033）
2. **障碍物分割**：以道路（10xx）、水系（11xx）、建设用地（20xx）为天然边界，在可置换地块上构建 Queen 邻接图
3. **连通分量**：BFS 提取连通分量（障碍物自然隔断）
4. **层次聚类细分**：对超过 30 个地块的大分量，使用 AgglomerativeClustering（Ward linkage + 连通性约束）细分为 ~20 个地块/组
5. **过滤**：移除过小碎片（< 3 个地块或 < 0.5 公顷）

### 2.2 地块组结果

| 镇 | 可置换地块数 | 地块组数 | 分配率 | 中位面积 |
|----|------------|---------|--------|---------|
| A（小镇） | 1,601 | 78 | 98% | 33 ha |
| B（中镇） | 2,669 | 132 | 99% | 28 ha |
| C（大镇） | 6,787 | 338 | 98% | 20 ha |

动作空间压缩比约 20 倍（地块数 → 地块组数）。

### 2.3 地块组邻接图

从地块级 Queen 邻接关系推导地块组邻接：若地块组 A 中任一地块与地块组 B 中任一地块相邻，则 A-B 相邻。中位邻居数约 4 个地块组。

---

## 3. MDP 形式化定义

### 3.1 基本设定

- **智能体角色**：区域土地整治资源配置决策者
- **决策粒度**：每步选择一个地块组进行投资
- **执行机制**：环境内部在选中地块组内自动运行连通性感知贪心置换引擎
- **设计理念**："宏观规划，微观执行"（Macroscopic Planning, Microscopic Execution）

### 3.2 状态空间 S

状态为一维连续向量，维度 = n_blocks × K_BLOCK + K_GLOBAL。

以 A 镇（78 个地块组）为例：78 × 17 + 9 = **1,335 维**。

#### 逐地块组特征（K_BLOCK = 17 维/地块组）

| 索引 | 符号 | 特征名称 | 计算方式 | 取值范围 |
|------|------|---------|---------|---------|
| 0 | f_slope_farm | 耕地平均坡度 | 面积加权均值，min-max 归一化 | [0, 1] |
| 1 | f_slope_forest | 林地平均坡度 | 面积加权均值，min-max 归一化 | [0, 1] |
| 2 | f_slope_gap | 耕-林坡度差 | (耕地均值 - 林地均值) / slope_range | [-1, 1] |
| 3 | f_best_gain | 最优单次置换收益 | (最陡耕地 - 最缓林地) / slope_range | [-1, 1] |
| 4 | f_slope_std | 耕地坡度标准差 | std / slope_range | [0, 1] |
| 5 | f_top_farm | 最陡耕地坡度 | min-max 归一化 | [0, 1] |
| 6 | f_bottom_forest | 最缓林地坡度 | min-max 归一化 | [0, 1] |
| 7 | f_farm_area | 可用耕地面积 | / max_block_area | [0, 1] |
| 8 | f_forest_area | 可用林地面积 | / max_block_area | [0, 1] |
| 9 | f_swap_pot | 置换潜力 | min(可用耕地数, 可用林地数) / 总地块数 | [0, 1] |
| 10 | f_invested | 已投入置换次数 | / swaps_per_step | [0, +∞) |
| 11 | f_compact | 形状紧凑度 | 等周商 4πA/P² | [0, 1] |
| 12 | f_area | 地块组总面积 | / max_block_area | [0, 1] |
| 13 | f_adj_inv | 相邻组已投资比例 | 已投资邻居数 / 邻居总数 | [0, 1] |
| 14 | f_adj_farm | 相邻组耕地面积 | 归一化 | [0, +∞) |
| 15 | f_self_farm | 本组当前耕地面积 | / max_block_area | [0, 1] |
| 16 | f_is_inv | 是否已被投资 | 布尔值 | {0, 1} |

**特征 13–16** 编码了跨地块组的空间关联信息，是百亩方形成机制的关键输入。

#### 全局特征（K_GLOBAL = 9 维）

| 索引 | 符号 | 特征名称 | 计算方式 |
|------|------|---------|---------|
| 0 | g_budget | 剩余预算比例 | 1 - step / max_steps |
| 1 | g_slope | 当前全局耕地均坡 | min-max 归一化 |
| 2 | g_cont | 当前全局连通度 | contiguity / 10 |
| 3 | g_step | 当前步数比例 | step / max_steps |
| 4 | g_slope_imp | 坡度累计改善率 | (initial - current) / \|initial\| |
| 5 | g_cont_imp | 连通度累计改善率 | (current - initial) / \|initial\| |
| 6 | g_baimu_cnt | 百亩方数量 | count / (n_blocks / 10) |
| 7 | g_baimu_frac | 百亩方面积占比 | 百亩方总面积 / 耕地总面积 |
| 8 | g_inv_frac | 已投资地块组比例 | 已投资数 / n_blocks |

### 3.3 动作空间 A

$$A = \{0, 1, \ldots, n\_blocks - 1\}$$

离散动作空间，每个动作对应选择一个地块组 ID。

**动作掩码**：地块组 b 可选当且仅当其内部同时存在可用耕地和可用林地：

$$\text{mask}(b) = (\text{farm\_avail}_b > 0) \wedge (\text{forest\_avail}_b > 0)$$

| 镇 | 动作空间大小 |
|----|------------|
| A（小镇） | 78 |
| B（中镇） | 132 |
| C（大镇） | 338 |

### 3.4 状态转移 T(s'|s, a)

状态转移是**确定性**的。选择动作 a = block_id 后：

1. 在 block_id 内执行**连通性感知贪心引擎**（最多 swaps_per_step = 5 次配对置换）
2. 更新全局状态（坡度、连通度、百亩方）

#### 连通性感知贪心引擎

对选中地块组内的可用地块：

**耕地移除**（farmland → forest）：
$$\text{score}_{\text{remove}}(i) = \text{slope}(i) - \delta \cdot \text{farmland\_nbr\_count}(i)$$
选择评分最高者（优先移除**陡坡 + 孤立**的耕地）。

**林地转入**（forest → farmland）：
$$\text{score}_{\text{convert}}(j) = \text{slope}(j) - \gamma \cdot \text{farmland\_nbr\_count}(j)$$
选择评分最低者（优先转入**缓坡 + 耕地邻居多**的林地）。

参数设定：γ = 1.0，δ = 0.5。

**置换条件**：仅当被移除耕地坡度 > 被转入林地坡度时执行，否则停止。

**跨地块组顺序依赖机制**：
- 投资地块组 A → 边界处新增耕地
- 相邻地块组 B 中林地的 farmland_nbr_count 增加
- 下次投资 B 时，边界林地优先被转为耕地（γ=1.0 的作用）
- A 和 B 的耕地跨边界连通 → 可能形成百亩方

当 γ = δ = 0 时，退化为纯坡度贪心（Paper 2 的 Greedy 基线）。

### 3.5 奖励函数 R(s, a, s')

每步奖励由 4 个分量线性组合：

$$R = w_{\text{slope}} \cdot \Delta_{\text{slope}} + w_{\text{cont}} \cdot \Delta_{\text{cont}} + w_{\text{baimu}} \cdot \Delta_{\text{baimu\_area}} + w_{\text{bonus}} \cdot N_{\text{new\_baimu}}$$

各分量定义：

| 分量 | 公式 | 含义 |
|------|------|------|
| Δ_slope | (prev_slope - cur_slope) / \|initial_slope\| | 坡度降低为正 |
| Δ_cont | (cur_cont - prev_cont) / \|initial_cont\| | 连通度增加为正 |
| Δ_baimu_area | (cur_area - prev_area) / initial_farm_area | 百亩方面积增长率 |
| N_new_baimu | max(0, cur_count - prev_count) | 新形成百亩方个数（仅奖励，不惩罚减少） |

权重设定：

| 参数 | 符号 | 值 | 说明 |
|------|------|------|------|
| w_slope | slope_weight | 2000 | 主导目标 |
| w_cont | cont_weight | 500 | 辅助目标 |
| w_baimu | baimu_weight | 500 | 百亩方面积增长 |
| w_bonus | baimu_bonus | 20 | 新增百亩方奖励 |

额外惩罚：若选中地块组实际完成 0 次置换（资源浪费），reward -= 1.0。

### 3.6 Episode 结构

| 参数 | 值 | 说明 |
|------|------|------|
| total_budget | 100 | 总置换预算（次） |
| swaps_per_step | 5 | 每步最多执行的置换次数 |
| max_steps | 20 | Episode 最大步数 = budget / swaps_per_step |
| 折扣因子 γ_RL | 0.99 | PPO 折扣因子 |

**终止条件**（满足任一）：
1. 步数达到 max_steps（20 步）
2. 所有地块组均无剩余置换潜力（耕地或林地耗尽）

### 3.7 百亩方定义与计算

**百亩方**：连通耕地总面积 ≥ 66,700 m²（= 100 亩 = 6.67 公顷）的连片耕地区域。

计算方法：
1. 在所有耕地地块上构建连通图（使用地块级 Queen 邻接关系，**跨地块组边界**）
2. BFS 遍历所有连通分量
3. 统计每个分量的总面积（各地块投影面积之和）
4. 面积 ≥ 阈值 → 计为一个百亩方

- 时间复杂度：O(N_parcels)
- 实测耗时：~1.3 ms/次（Township A, 1601 parcels）
- 调用频率：每步调用一次

### 3.8 连通度指标（Contiguity）

$$\text{Contiguity} = \frac{\sum_{i \in \text{farmland}} \text{farmland\_nbr\_count}(i)}{|\text{farmland}|}$$

即每个耕地地块的耕地邻居数之和除以耕地总数。值越大表示耕地越连片。

---

## 4. 策略网络架构

### 4.1 网络结构：EntityScoringPolicy（维度无关架构）

采用 Per-Entity Scoring 架构，与 Paper 1 的 ParcelScoringPolicy 同构：

**Scorer 网络**（共享权重，逐实体打分）：
$$\text{Scorer}: \mathbb{R}^{K\_BLOCK + K\_GLOBAL} \rightarrow \mathbb{R}^{1}$$
$$\text{结构}: (26) \rightarrow [128] \rightarrow \tanh \rightarrow [64] \rightarrow \tanh \rightarrow (1)$$

- 输入：每个地块组的 17 维特征 + 9 维全局特征拼接 = 26 维
- 输出：1 个 logit 值
- 对所有地块组共享同一组权重
- 所有 logits 经 softmax + action masking → 动作概率分布

**Value 网络**（状态价值估计）：
$$V: \mathbb{R}^{K\_GLOBAL} \rightarrow \mathbb{R}^{1}$$
$$\text{结构}: (9) \rightarrow [64] \rightarrow \tanh \rightarrow [32] \rightarrow \tanh \rightarrow (1)$$

- 仅接收全局特征，不依赖地块组数量
- 输出标量状态价值

### 4.2 参数统计

| 组件 | 结构 | 参数量 |
|------|------|--------|
| Scorer | (26)→128→64→1 | 12,097 |
| Value | (9)→64→32→1 | 2,433 |
| **总计** | | **14,530** |

### 4.3 维度无关性

同一套训练权重可直接应用于不同地块组数量的镇：
- 训练时 N=78（A 镇），可直接迁移到 N=132（B 镇）或 N=338（C 镇）
- Scorer 逐实体打分，Value 只看全局特征，均不依赖 N

### 4.4 与 Paper 1 的对比

| | Paper 1 | Paper 3 |
|--|---------|---------|
| 评分对象 | 地块（parcel） | 地块组（block） |
| 实体特征维度 | 6（PoC）/ 10（real） | 17 |
| 全局特征维度 | 8 | 9 |
| Scorer 输入 | 14 / 18 | 26 |
| Scorer 隐层 | [64, 32] | [128, 64] |
| Value 隐层 | [64, 32] | [64, 32] |
| 动作空间 | ~1,601 parcels | ~78 blocks |
| 总参数量 | ~3,500 | 14,530 |

---

## 5. 训练算法

### 5.1 算法：MaskablePPO

基于 sb3-contrib 的 Maskable Proximal Policy Optimization。

### 5.2 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| learning_rate | 1×10⁻³ | 学习率 |
| n_steps | 512 | Rollout buffer 长度 |
| batch_size | 256 | Mini-batch 大小 |
| n_epochs | 10 | 每次更新的 epoch 数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE λ |
| clip_range | 0.2 | PPO 裁剪范围 |
| ent_coef | 0.01 | 熵系数（低值，78 个动作需尖锐策略） |
| vf_coef | 0.5 | 价值函数系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| total_timesteps | 200,000 | 总训练步数 |

### 5.3 超参调优关键发现

1. **ent_coef**: 0.05 导致策略退化（clip_fraction=0），0.01 有效。78 个动作的空间下高熵系数会将策略推向均匀分布。
2. **learning_rate**: 1e-3 优于 3e-4，因为动作空间较小，可承受较大学习率。
3. **reward weights**: slope_weight 必须主导。baimu_weight=2000 + slope_weight=1000 导致坡度恶化；调整为 slope=2000, baimu=500 后两个目标均改善。

---

## 6. 基线方法

所有基线使用同一个连通性感知贪心引擎（γ=1.0, δ=0.5），仅地块组选择策略不同：

| 方法 | 选择策略 | 说明 |
|------|---------|------|
| Greedy-Global | 忽略地块组，全局排序 | Paper 2 的 Greedy 基线适配 |
| Greedy-Sequential | 按耕-林坡度差降序 | 优先投资坡度差最大的组 |
| Random-Block | 随机选择（5 seeds） | 随机基线 |
| Round-Robin | 固定顺序轮询 | 均匀分配基线 |

---

## 7. 评价指标

| 指标 | 公式/定义 | 方向 |
|------|----------|------|
| 坡度变化率 | 100 × (final_slope - initial_slope) / \|initial_slope\| | 越负越好 |
| 连通度变化 | final_contiguity - initial_contiguity | 越正越好 |
| 百亩方数量变化 | final_count - initial_count | 越正越好 |
| 百亩方面积变化 | (final_area - initial_area) / 10000 (ha) | 越正越好 |
| 预算使用量 | 实际完成的置换次数 | 信息性指标 |
