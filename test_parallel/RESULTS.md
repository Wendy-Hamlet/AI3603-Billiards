# 并行评估测试结果

**测试时间**: 2024-12-25  
**测试分支**: merge_basic  
**测试环境**: 100 CPU cores, 80 workers

---

## 测试配置

| 参数 | 值 |
|------|-----|
| 测试局数 | 1000 |
| 并行进程 | 80 |
| 超时时间 | 600s |
| Agent 来源 | `time_limit_mcts/agent.py` |

---

## 测试结果

### 1. merge_basic (NewAgent) vs BasicAgent

| 指标 | 值 |
|------|-----|
| **总局数** | 1000 |
| **有效局数** | 1000 |
| **错误/超时** | 0 |
| **总耗时** | 22.5 分钟 |
| **平均速度** | 0.74 局/秒 |
| **平均击球数** | 28.6 |

| Agent | 胜场 | 胜率 |
|-------|------|------|
| **NewAgent (merge_basic)** | 904 | **90.40%** |
| BasicAgent | 80 | 8.00% |
| 平局 | 16 | 1.60% |

**结论**: NewAgent 对 BasicAgent 有压倒性优势，胜率超过 90%。

---

### 2. merge_basic (NewAgent) vs BasicAgentPro

| 指标 | 值 |
|------|-----|
| **已完成局数** | 990/1000 |
| **状态** | 最后 10 局卡住（可能是 Pro 模拟超时） |

| Agent | 胜场 | 胜率 (基于 990 局) |
|-------|------|------|
| **NewAgent (merge_basic)** | 556 | **56.2%** |
| BasicAgentPro | 434 | 43.8% |
| 平局 | 0 | 0% |

**结论**: NewAgent 对 BasicAgentPro 有一定优势，胜率约 56%，符合预期（队友报告的 50-60%）。

---

## 各分支胜率汇总（来自队友报告）

| 分支 | 对手 | 胜率 |
|------|------|------|
| time_limit_mcts | BasicAgent | 93.75% |
| robust | BasicAgent | 95% |
| merge_basic | BasicAgent | **90.40%** (本次测试) |
| merge_basic | BasicAgentPro | **56.2%** (本次测试) |

---

## 备注

1. **BasicAgent**: 基于贝叶斯优化的智能 Agent（30 次迭代）
2. **BasicAgentPro**: 基于 MCTS 的进阶 Agent（50 次模拟）
3. **NewAgent (merge_basic)**: 融合了 robust 分支的 MCTS 和 BasicAgentPro 策略

---

## 测试命令

```bash
# vs BasicAgent
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 80 \
    --agent_a merge_basic --agent_b basic

# vs BasicAgentPro
python test_parallel/parallel_evaluate.py \
    --num_games 1000 --num_workers 80 \
    --agent_a merge_basic --agent_b pro
```

