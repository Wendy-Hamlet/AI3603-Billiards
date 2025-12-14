# NewAgent 实现说明

## 核心设计理念

针对BasicAgent在噪声环境下偶尔误打黑8的问题，NewAgent采用**混合智能体架构**，结合战略规划和噪声鲁棒性评估，实现更稳定的决策。

---

## 主要改进点

### 1. 三层战略系统

根据当前局面动态选择不同策略：

| 战略模式 | 触发条件 | 搜索空间 | 特点 |
|---------|---------|---------|------|
| **aggressive（进攻）** | 前期优势/清台后期领先 | V0: 4.0-8.0 m/s | 高速击球，快速清台 |
| **defensive（防守）** | 劣势/均势/接近黑8 | V0: 0.5-3.0 m/s | 低速精准，避免失误 |
| **final_eight（黑8）** | 己方球全清，只剩黑8 | V0: 1.0-4.0 m/s | 极度保守，多次验证 |

**战略判断逻辑**：
```python
# 剩余球数统计
remaining_own = 我方未进袋的目标球
remaining_enemy = 对方未进袋的目标球

# 黑8阶段：绝对优先
if '8' in my_targets:
    return 'final_eight'

# 前期（≥5球）：优势进攻
elif len(remaining_own) >= 5:
    if remaining_own <= remaining_enemy - 2:
        return 'aggressive'
    else:
        return 'defensive'

# 后期（2-4球）：谨慎评估
elif len(remaining_own) >= 2:
    if remaining_own <= remaining_enemy:
        return 'aggressive'
    else:
        return 'defensive'

# 接近黑8（1球）：极度保守
else:
    return 'defensive'
```

---

### 2. 噪声鲁棒性评估（核心创新）

针对环境噪声导致的误打黑8问题，引入**蒙特卡洛采样**评估动作稳定性：

```python
def _evaluate_action_robustness(action, n_samples=5):
    """
    对候选动作添加噪声，模拟n_samples次，评估：
    - mean_score: 平均得分（期望表现）
    - std_score: 标准差（稳定性）
    - min_score: 最坏情况（鲁棒性）
    """
    scores = []
    for _ in range(n_samples):
        # 添加与环境一致的高斯噪声
        noisy_action = {
            'V0': action['V0'] + N(0, 0.1),
            'phi': action['phi'] + N(0, 0.1),
            'theta': action['theta'] + N(0, 0.1),
            'a': action['a'] + N(0, 0.003),
            'b': action['b'] + N(0, 0.003)
        }
        # 物理模拟
        score = simulate_and_score(noisy_action)
        scores.append(score)
    
    return mean(scores), std(scores), min(scores)
```

**应用场景**：
- 黑8阶段：每个候选动作必须通过噪声测试
- 综合评分：`robust_score = 0.3*base + 0.5*mean - 2*std`
- 安全过滤：`if min_score < -50: reject`

---

### 3. 黑8阶段专项优化

黑8是最容易误打的关键阶段，采用三重保护：

#### (1) 搜索空间限制
```python
'final_eight': {
    'V0': (1.0, 4.0),      # 限制低速（BasicAgent: 0.5-8.0）
    'theta': (30, 80),     # 限制垂直角度（避免跳球）
    'a': (-0.15, 0.15),    # 限制偏移（BasicAgent: -0.5~0.5）
    'init_points': 20,     # 增加初始采样（BasicAgent: 20）
    'n_iter': 15           # 增加优化轮次（BasicAgent: 10）
}
```

#### (2) 动作内噪声测试
在贝叶斯优化过程中，对高分候选动作进行3次噪声采样：
```python
if strategy == 'final_eight' and base_score > 50:
    mean, std, min_score = evaluate_robustness(action, n_samples=3)
    robust_score = 0.3*base + 0.5*mean - 2*std
    if min_score < -50:
        robust_score -= 100  # 严重惩罚
```

#### (3) 最终验证 + 备用方案
选定动作后，再进行8次噪声采样验证：
```python
if strategy == 'final_eight':
    mean, std, min_score = evaluate_robustness(action, n_samples=8)
    if min_score < -50 or std > 30:
        # 噪声不稳定 → 启动备用超保守决策
        return fallback_defensive_decision()
```

备用决策特点：
- V0限制在0.8-2.5 m/s（极低速）
- theta限制在40-85度（强制垂直击打）
- 偏移限制在±0.1（几乎中心击球）
- 搜索次数：25+20=45次（确保找到安全方案）

---

### 4. 安全过滤器

在贝叶斯优化评分前，先进行快速安全检查：

```python
def _is_safe_action(action, balls, my_targets):
    """
    检查是否可能误打黑8（清台前）
    """
    if '8' not in my_targets:  # 黑8还不是目标
        cue_pos = 白球位置
        eight_pos = 黑8位置
        dist_to_eight = ||cue_pos - eight_pos||
        
        # 找最近的己方目标球
        min_dist_to_target = min(||cue_pos - target_pos||)
        
        # 风险判定：黑8比目标球近 且 速度大
        if dist_to_eight < 0.8 * min_dist_to_target and V0 > 5.0:
            return False  # 高风险，直接惩罚-200
    
    return True
```

---

## 完整决策流程

```
1. 战略评估 → 确定 strategy ∈ {aggressive, defensive, final_eight}
                ↓
2. 配置搜索空间 → 根据 strategy 调整 pbounds, init_points, n_iter
                ↓
3. 贝叶斯优化（每次评分时）：
   - 安全过滤（快速几何检查）
   - 物理模拟（2秒超时保护）
   - 噪声评估（仅黑8阶段，3次采样）
                ↓
4. 选定最佳动作 → best_params
                ↓
5. 最终验证（仅黑8阶段）：
   - 8次噪声采样
   - 若不稳定 → fallback_defensive_decision()
                ↓
6. 返回动作
```

---

## 关键参数说明

### 噪声采样次数（n_samples）

| 阶段 | 次数 | 原因 |
|------|------|------|
| 黑8优化内测试 | 3次 | 平衡准确性和速度（每个候选动作都要测） |
| 黑8最终验证 | 8次 | 高置信度（只验证最终动作） |
| 其他阶段 | 0次 | 节省计算时间 |

### 搜索预算（时间控制）

| 战略 | init_points | n_iter | 总评估次数 | 预计耗时 |
|------|-------------|--------|-----------|---------|
| aggressive | 15 | 8 | 23 | ~15-20秒 |
| defensive | 12 | 10 | 22 | ~15-20秒 |
| final_eight | 20 | 15 | 35 | ~40-50秒（含噪声测试） |
| 备用方案 | 25 | 20 | 45 | ~30-40秒 |

---

## 与BasicAgent的差异对比

| 特性 | BasicAgent | NewAgent |
|------|-----------|----------|
| 搜索空间 | 固定（全局） | 动态（根据战略调整） |
| 搜索次数 | 固定（20+10） | 动态（12~45） |
| 噪声处理 | 可选（训练时开关） | 强制评估（黑8阶段） |
| 黑8保护 | 无特殊处理 | 三重保护（限制+测试+备用） |
| 安全过滤 | 无 | 几何快速检查 |
| 决策时间 | 15-25秒 | 15-50秒（黑8阶段较慢） |

---

## 预期效果

### 优势
1. **黑8误打率大幅降低**：三重保护机制确保噪声鲁棒性
2. **战略多样性**：根据局面动态调整风格（进攻/防守）
3. **搜索效率提升**：缩小搜索空间后，单位时间内探索更高质量区域

### 劣势
1. **黑8阶段决策慢**：噪声评估增加40-50%耗时
2. **防守模式激进不足**：为保稳定牺牲部分进攻性

### 胜率预估
- vs BasicAgent（无噪声）：45-55%（持平或略优）
- vs BasicAgent（有噪声）：**60-70%**（黑8误打减少带来的优势）

---

## 使用方法

### 1. 替换agent.py
将生成的`agent.py`替换项目中的原文件。

### 2. 运行评估
```bash
cd /your/project/path
conda activate poolenv
python evaluate.py
```

### 3. 调整参数（可选）

如果想进一步优化，可以修改：

```python
# 在 NewAgent.__init__() 中

# 调整战略配置
self.strategies['final_eight']['init_points'] = 25  # 增加搜索（更慢但更准）
self.strategies['aggressive']['pbounds']['V0'] = (5.0, 8.0)  # 更激进

# 调整噪声测试次数
# 在 decision() 中，搜索第262行
n_samples=3  # 改为5（更准确但更慢）

# 在 decision() 中，搜索第290行
n_samples=8  # 改为12（最终验证更严格）
```

---

## 预期改进方向

如果有更多时间，可以考虑：

1. **并行化噪声采样**：使用多进程加速（可提速3-4倍）
2. **学习型战略网络**：训练小型神经网络预测最佳战略
3. **对手建模**：根据对手风格调整策略（进攻型/防守型）
4. **轨迹规划**：考虑2-3步后的局面（目前只看单步）

---

## 总结

NewAgent通过**战略分层 + 噪声鲁棒评估 + 黑8专项保护**，解决了BasicAgent在噪声环境下的不稳定问题，尤其是黑8误打。核心思想是：**用略微增加的计算时间，换取决策的稳定性和安全性**。

预计在120局评估中，相比BasicAgent能够减少5-10局因黑8误打导致的失败，胜率提升至60-70%。
