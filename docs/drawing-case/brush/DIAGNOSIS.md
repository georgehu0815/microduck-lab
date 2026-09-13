# Drawing PPO 历史诊断

Author: George Hu  
Date: 2026-09-11

## 结论

现有两次 drawing PPO run 都从较好的 BC/DAgger policy 发生退化。Checkpoint
测量确认 128 次 update 后存在 cumulative policy drift；缺少 BC reference
anchor 与该退化一致，是当前高优先级 causal hypothesis，但尚无 single-factor
ablation 证明它是主要原因。另有一个已证明的训练 bug：每次 BC/DAgger
结束后，代码把 `log_std` 无条件重置为 `DEFAULT_INITIAL_STD=0.02`，覆盖 CLI
配置值。

本次历史修复只处理 std reset，不修改 logging、reward、environment、PPO 参数或
其他历史行为。独立的 `brush-color-v2-03` 最终验收现已完成：PPO 与 matching BC
都在 actual Studio API 的 10 组配置 x 4 seeds 中 40/40 通过。Brush 使用 separate
new design；matching BC 同样通过，所以该结果既不是上述历史退化原因的 causal
proof，也不支持 PPO 优于 BC，更不能与这两次 drawing run 混为同一训练结果。当前
最终 hardened evaluator 的 PPO 与 matching BC 重评估均已完成，创建时间分别为
`2026-09-11T22:20:38Z` 和 `2026-09-11T22:24:48Z`；两者均为 40/40、
`training_pipeline_source_match=false`、`provenance_errors=[]`。这些报告已包含 checkpoint
内嵌 hash 和六份 source archive 完整性检查，并与当前 evaluator digest 一致。

## Forensic measurements

以下数值来自只读加载现有 NPZ、SB3 checkpoint、CSV 和 evaluation artifact；
未启动训练。

| Measurement | `drawing-pilot-01` | `drawing-refinement-02` |
|---|---:|---:|
| BC mean coverage | 0.334233 | 0.672136 |
| PPO mean coverage | 0.158828 | 0.242315 |
| BC teacher-state action MSE | 2.9675625e-6 | 3.220124e-7 |
| PPO teacher-state action MSE | 1.358264e-4 | 5.4708868e-5 |
| MSE increase | 45.77x | 169.90x |
| BC-to-PPO action drift RMS | 0.011624 rad | 0.007282 rad |
| KL(BC || PPO), teacher observations, mean | 2.534111 | 0.994400 |
| Logged PPO `approx_kl`, mean | 0.031267 | 0.004286 |
| Logged updates above `target_kl=0.015` | 98.43% | 0% |
| Logged value loss, mean | 192.218 | 344.628 |

Refinement policy 在 teacher observations 上的主要 per-action drift RMS：

- `neck_pitch`: 0.016205 rad
- `head_yaw`: 0.010617 rad
- `head_roll`: 0.009130 rad
- `left_ankle`: 0.007803 rad

Teacher dataset 的 target-error norm p95 约为 2.1 mm。现有 DAgger recovery
states 的 p95 达 17-58 mm；pilot 最大值约 149 mm。Teacher joint-velocity
p95 约 0.092 rad/s，pilot DAgger 最大值约 7.51 rad/s。因此固定 observation
scaling 在 recovery distribution 上并非归一化。

Action contract 是 15 个 absolute normalized actuator offsets。Teacher 的八个
leg action 维度恒为零；`std=0.02` 仍会在全部 action 维度加入约 0.02 rad
（1.15 degree）探索噪声。原计划的 `std=0.005` 约为 0.29 degree。

## Proven bug 与 causal boundary

`policy_kwargs(initial_std)` 正确用 CLI 参数初始化 `log_std`，但
`behavior_clone()` 在所有 BC 和 DAgger round 后执行：

```python
model.policy.log_std.fill_(math.log(DEFAULT_INITIAL_STD))
```

BC loss 使用 deterministic actor mean，不训练 `log_std`。因此该 reset 没有
必要，并直接破坏非默认 std 配置。这一 std reset bug 由代码路径和历史
checkpoint 一致证明。

`drawing-refinement-02` 的历史 metadata 记录 `initial_std=0.005`，表达的是
**intended configuration，不是 actual PPO starting std**。读取现有
`bc_policy.zip` 后，15 个维度的实际 std 全部为 0.02；PPO 最终 std 仍约为
0.02。`drawing-pilot-01` 本来就配置 0.02，因此不受 metadata discrepancy
影响，但仍存在累计 PPO drift。

两份历史 `eval.json` 都记录 `environment_source_match=false`，说明 evaluation
environment 与 training source hash 不同。BC/PPO 使用相同 evaluator，所以同一
run 内的相对退化仍是有效证据；但不能把所有绝对指标精确归因到当前 source。

当前 brush evaluator 的 provenance 规则更细：environment、reference、actor 和
base source 必须匹配；ONNX bytes 在评估开始时一次性读取并保持 immutable；原训练
pipeline 与其余五份训练 source 必须分别由 run 中六份归档文件匹配内嵌 hash；实际
checkpoint hash 还必须同时匹配 sidecar 和 ONNX 内嵌 metadata。只修改 evaluator
pipeline 是允许的，此时报告以 `training_pipeline_source_match=false` 明示训练脚本
与当前评估脚本不同，并不要求仅因此重新训练。

这些是训练后加入的 hardened evaluation gate。归档的 final03 旧训练 pipeline 会在
加载数据时验证 dataset 与六项 source hash，并保存六份 source，但不包含当前反复执行
的全生命周期 `assert_sources`。因此不能倒推声称 final03 训练时已经使用该新增 guard。
该边界也不改变上述两个历史 drawing artifact 的来源结论。

## Observed 与 hypothesis

**Observed**

- Teacher 完成 32 秒轨迹，coverage 和 precision 均为 1.0。
- 两个 PPO checkpoint 相对各自 BC baseline 都降低 coverage、precision 和
  mean return。
- Refinement 即使每次 logged KL 都低于 target，最终仍产生约 0.994 的
  cumulative KL。
- Refinement 历史 metadata 的 `initial_std=0.005` 与 checkpoint 中实际
  `std=0.02` 不一致。

**Attribution**

- Std mismatch 的直接原因是 unconditional reset。
- PPO 相对 BC 的 measured drift 直接表现为 teacher/DAgger states 上 action
  error 和 cumulative KL 明显增加；这是退化的量化描述，不单独证明 drift
  由某一个 training choice 引起。

**Hypothesis，未在本次修复**

- 缺少 BC anchor 与 measured cumulative drift 一致，但尚无 single-factor
  causal ablation，不能称为已证明的 primary root cause。
- Step-local reward、短 rollout 和未预训练 critic 可能共同允许 PPO 优化离开
  完整绘图 acceptance objective。
- DAgger recovery-state scale shift 增加 actor extrapolation 难度。

这些 hypothesis 需要专门 ablation 验证。当前 brush PPO 与 matching BC 都已在
40-episode final gate 中通过，但它采用 separate new design，且没有显示 PPO 相对
BC 的改进，因此不能作为这些历史 causal hypothesis 的证明；本次修复也不改变它们。
