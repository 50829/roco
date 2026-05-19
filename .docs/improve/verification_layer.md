# 通用 Verification Layer 优化记录

## 1. 背景与问题判断

在 Sort、Pack、Rope 三类任务中，LLM 规划经常出现两类问题：

1. **语义上不合法**：动作名错误、目标错误、搬错物体、阶段不对、并发数量过多等。
2. **符号上看似合理但物理上容易失败**：多个机器人同时进入共享区域、等待机器人仍然持物挡路、绳子单端移动等。

之前主要依赖 prompt 约束和环境反馈。问题是：

- prompt 不能保证模型一定遵守规则；
- parser/RRT 之后才发现错误，代价高；
- 不同任务的规则分散在各自 prompt 和 `get_task_feedback()` 中；
- Sort 已经有一部分 legal action 校验，但它偏 Sort 专用，不能直接覆盖 Pack/Rope；
- 失败后 LLM 容易重复输出同样的非法动作。

因此决定加入一层更强的 **verification layer**，在 parser/RRT 之前先做硬校验。

目标不是完全替代 RRT，而是提前拦截明显不合法或高风险的计划：

```text
LLM output
→ generic verification layer
→ task-specific semantic verifier
→ parser
→ feedback_manager
→ RRT
```

---

## 2. 方案选择过程

### 2.1 是否使用另一个 LLM agent 做 verifier

一开始考虑过使用“校验 agent”审查规划结果。但最终没有采用 LLM verifier，原因是：

- LLM verifier 也可能误判；
- 会增加推理延迟；
- 两个 LLM 可能互相给出看似合理但仍不满足物理约束的解释；
- 当前任务的规则大多可以确定性检查。

所以最终采用：

```text
通用 deterministic verifier 框架
+ 每个任务自己的 task-specific verification hook
```

### 2.2 为什么不能一套规则硬套所有任务

不同任务的动作语义差异很大：

- Sort 是离散 handoff 任务；
- Pack 有 bin slot、held object、PLACE 串行约束；
- Rope 有两端同步、PICK/PUT 阶段机、路径高度和绳长约束。

因此通用层只做共性检查；任务强规则由 env hook 自己实现。

---

## 3. 实现过程

### 3.1 修改 `plan_prompter.py`

文件：

```text
code/prompting/plan_prompter.py
```

把原来的 `_validate_against_legal_actions()` 扩展为通用 verification layer。

现在它会检查：

```text
1. response 是否包含每个机器人 exactly one action；
2. action name 是否属于 env.get_allowed_action_names()；
3. 如果 env 提供 get_legal_actions(obs)，则 action 必须精确在 legal actions 里；
4. 是否重复本轮 forbidden action；
5. 是否多个机器人 PICK 同一物体；
6. 是否多个机器人 PLACE/PUT 到同一目标；
7. 是否超过 env.get_max_parallel_actions(obs)；
8. 是否通过 env.verify_plan_semantics(obs, actions)。
```

新增的 env hook 包括：

```python
env.get_allowed_action_names()
env.get_max_parallel_actions(obs)
env.verify_plan_semantics(obs, actions)
```

同时修复了一个策略问题：

如果失败原因是：

```text
Too many non-WAIT actions
```

则不把每个单独动作都加入 forbidden。因为这类错误代表“组合不安全”，不代表单个动作本身错误。这样 fallback 仍然可以尝试：

```text
一个 active action + 其他机器人 WAIT
```

---

### 3.2 Sort 接入强校验

文件：

```text
code/rocobench/envs/task_sort.py
```

新增：

```python
get_allowed_action_names()
get_max_parallel_actions()
verify_plan_semantics()
```

Sort 当前强规则：

```text
- 只允许 PICK / WAIT；
- 每轮最多 1 个非 WAIT action；
- 非 WAIT action 必须在 current legal actions 中；
- 任务未完成时不能全 WAIT；
- 不能移动已经 done 的 cube；
- 必须遵守 directed handoff route；
- blue_square -> panel2；
- pink_polygon -> panel4；
- yellow_trapezoid -> panel6；
- Alice/Bob handoff 使用 panel3；
- Bob/Chad handoff 使用 panel5。
```

同时在 Sort prompt 中补充：

```text
For reliability, use exactly one non-WAIT action per round.
Never make two robots move in the same round.
```

这样 LLM 即使输出多个 legal actions，也会在进入 parser/RRT 前被 verifier 拦截。

---

### 3.3 Pack 接入任务级校验

文件：

```text
code/rocobench/envs/task_pack.py
```

新增：

```python
get_allowed_action_names()
get_max_parallel_actions()
verify_plan_semantics()
```

Pack 当前规则：

```text
- 只允许 PICK / PLACE / WAIT；
- 双空手时最多允许 2 个 active actions；
- 有任意机器人 holding item 时最多允许 1 个 active action；
- 不允许两个机器人同时 PLACE；
- 不允许 PLACE + PICK 混合同一轮；
- holding 状态必须和动作一致；
- PLACE 必须放到显式 bin slot；
- 不能 PLACE 到 occupied slot；
- 有机器人 holding 时，必须有一个 holding robot PLACE，另一个 WAIT/retreat。
```

这个 hook 是为了防止之前 Pack 中出现的问题：

```text
Bob WAIT 但仍 holding milk，Alice 搬 cereal 时发生 cereal-Bob collision。
```

虽然 Pack 后续没有继续深测，但现在至少能在语义层面提前拦截明显错误的并发和 holding 状态不一致问题。

---

### 3.4 Rope 接入阶段机校验

文件：

```text
code/rocobench/envs/task_rope.py
```

新增：

```python
get_allowed_action_names()
get_max_parallel_actions()
verify_plan_semantics()
```

Rope 当前规则：

```text
- 只允许 PICK / PUT / WAIT；
- 禁止 LIFT / PLACE / MOVE / LOWER / RAISE / DROP；
- 每个动作必须包含 PATH；
- 空手阶段：Alice PICK rope_front_end，Bob PICK rope_back_end；
- 单端 holding 阶段：holding robot WAIT，empty robot PICK 另一端；
- 双端 holding 阶段：两边必须同时 PUT；
- 不允许 PICK + PUT 混合同一轮；
- holding 哪一端，就 PUT 哪一端。
```

这把 Rope 从纯 prompt 约束推进到明确阶段机约束。

---

## 4. 实现中遇到的问题与再次尝试

### 4.1 问题：Sort 原本已经有 legal action 校验，如何不破坏它

原逻辑里：如果 env 提供 `get_legal_actions()`，LLM 输出必须精确属于 legal actions。

这对 Sort 很重要，不能删掉。于是通用 verifier 采用兼容方式：

```text
如果 env 有 get_legal_actions(obs)：继续强制 action in legal_actions；
如果 env 没有：只使用通用 action name / duplicate / max_parallel / semantic hook。
```

这样 Sort 保持强约束，Pack/Rope 也能获得基础校验。

### 4.2 问题：并发超限是否应该 ban 单动作

第一次设计时，任何校验失败都可能触发 ban。后来分析发现这是错误的：

```text
Alice action 合法
Bob action 合法
但 Alice+Bob 同时动不合法
```

这种情况下不应该 ban Alice 或 Bob 的单独动作。

因此修改为：

```text
如果失败原因是 Too many non-WAIT actions，不 ban 单独 action。
```

这样 deterministic fallback 可以继续尝试单机器人子计划。

### 4.3 问题：通用 duplicate target 检查如何兼容 PUT / PLACE

Sort 使用 `PICK ... PLACE ...`，Pack 使用 `PLACE obj slot PATH`，Rope 使用 `PUT obj groove PATH`。

所以通用 verifier 中分别解析：

```text
PICK ... PLACE ...  → target 来自 PLACE 后面
PLACE obj target PATH → target 是第二个 token
PUT obj target PATH   → target 是第二个 token
```

这样能覆盖三类任务的重复目标检测。

### 4.4 问题：测试环境 MuJoCo GL 报错

轻量实例化环境时遇到：

```text
gladLoadGL error
X11: The DISPLAY environment variable is missing
```

这是渲染后端问题，不是 verifier 逻辑问题。

再次尝试时加入：

```bash
MUJOCO_GL=egl
```

之后 Sort 轻量检查可以运行。

---

## 5. 当前验证结果

### 5.1 编译检查

执行：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code
python -m py_compile prompting/plan_prompter.py rocobench/envs/task_sort.py rocobench/envs/task_pack.py rocobench/envs/task_rope.py
```

结果：通过。

### 5.2 Sort 轻量 verifier 检查

执行了一个轻量脚本：

- 初始化 Sort 环境；
- 打印 legal actions；
- 检查 recommended plan 是否通过 verifier；
- 构造一个 3 个机器人同时 active 的非法计划；
- 检查是否被 verifier 拦截。

结果：

```text
recommended valid: (True, 'OK')
multi active valid: False
```

非法并发被拦截，错误信息为：

```text
Too many non-WAIT actions: 3.
This task allows at most 1 non-WAIT action(s) per round.
Use WAIT for the other robots.
```

这说明 Sort 的单 active 约束已经在 parser/RRT 前生效。

---

## 6. 当前最终状态

目前 verification layer 已经完成基础接入：

```text
通用校验：plan_prompter.py
Sort 强语义校验：task_sort.py
Pack 状态/slot/holding 校验：task_pack.py
Rope 阶段机校验：task_rope.py
```

当前最直接的预期收益：

```text
- Sort：减少 LLM 多机器人并行动作导致的碰撞/RRT 失败；
- Pack：提前阻止 holding 状态和 PLACE/PICK 混合错误；
- Rope：提前阻止阶段错误和非法动作名。
```

但还没有做完整批量实验。下一步应由实验运行验证：

```text
1. Sort 单 run 是否正常；
2. Sort 多 run 成功率是否高于旧的 sort_legal_plan_test；
3. Pack/Rope 是否因为 verifier 过严导致无法推进；
4. 若出现过严，则针对对应 task 放宽 hook，而不是削弱通用层。
```

---

## 7. 建议的 Sort 测试命令

单次测试：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code

export ROCO_DATA_DIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data
export MPLCONFIGDIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data/.matplotlib
export MUJOCO_GL=egl
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export ROCO_LLM_MODEL=llama3.3:70b

python run_dialog.py \
  --task sort \
  --run_name sort_verifier_test \
  --num_runs 1 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

多次测试：

```bash
python run_dialog.py \
  --task sort \
  --run_name sort_verifier_test_10runs \
  --num_runs 10 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

更长步数测试：

```bash
python run_dialog.py \
  --task sort \
  --run_name sort_verifier_test_t15 \
  --num_runs 5 \
  --tsteps 15 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

---

## 8. 后续改进方向

1. **IK precheck**：只对 LLM 输出动作和 fallback 候选做 IK 预检查，提前过滤明显 IK 不可解动作。
2. **Sort symbolic planner**：Sort 规则离散清晰，后续可以直接 BFS/A* 生成动作序列，LLM 只做解释或兜底。
3. **Pack retreat verifier**：进一步检查 WAIT robot 是否 holding bulky object 且挡住 active PLACE corridor。
4. **Rope geometry verifier**：检查双臂高度差、绳端距离变化、是否越过墙顶。
5. **失败分类**：区分 semantic invalid、IK temporary failure、collision combination failure，避免过度 ban 必经动作。
