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

---

## 9. Sort Safe Parallel 实验与继续改进记录

### 9.1 为什么继续改

在串行 verifier 版本中，Sort 的稳定性明显提高：

```text
sort_verifier_test: 10 / 10 成功
平均成功步数约 6.1
```

但是该版本每轮基本只有一个机械臂工作，效率偏低。为了提升效率，继续尝试开放有限并行。

目标不是回到任意并行，而是：

```text
只允许低风险并行；
中间机器人 Bob 仍保持串行；
两端机器人 Alice 和 Chad 在工作区不重叠时可以同时行动。
```

---

### 9.2 Safe Parallel 的实现思路

对 Sort 加入保守并行规则：

```text
- 每轮最多 2 个 active actions；
- 只允许 Alice + Chad 并行；
- Bob 是中间机器人，不能和任何人并行；
- 并行动作不能移动同一个 cube；
- 并行动作不能放到同一个 target；
- 两个动作的 panel footprint 不能重叠。
```

新增/修改的核心函数在：

```text
code/rocobench/envs/task_sort.py
```

包括：

```python
get_max_parallel_actions()
_parse_sort_action()
_panel_index()
_sort_action_footprint()
_sort_parallel_compatible()
```

其中 `_sort_action_footprint()` 用来粗略估计一个动作经过的 panel corridor，例如：

```text
PICK object on panel1 PLACE panel3 -> footprint {1,2,3}
PICK object on panel7 PLACE panel5 -> footprint {5,6,7}
```

如果 Alice 和 Chad 的 footprint 不重叠，就认为可以并行。

---

### 9.3 推荐计划从单动作 greedy 改成组合选择

原来的 `get_recommended_plan()` 是：

```text
从所有候选动作中选分数最高的一个动作。
```

这会导致即使 Alice + Chad 可以安全并行，推荐计划也只输出一个动作。

因此改成：

```text
枚举最多 2 个候选动作组合；
过滤掉不兼容组合；
选择总分最高的组合。
```

这样在典型初始状态下可以推荐：

```text
EXECUTE
NAME Alice ACTION PICK pink_polygon PLACE panel3
NAME Bob ACTION WAIT
NAME Chad ACTION PICK blue_square PLACE panel5
```

轻量验证结果显示该推荐计划能通过 verifier。

---

### 9.4 Safe Parallel 实验结果

用户运行了：

```text
sort_safe_parallel_test
```

结果：

```text
10 runs
7 成功
3 失败
成功率 70%
成功样本平均步数约 4.43
```

与串行 verifier 对比：

```text
sort_verifier_test:
10 / 10 成功
平均 6.1 步

sort_safe_parallel_test:
7 / 10 成功
平均成功 4.43 步
```

说明：

```text
safe parallel 明显提升了成功样本的效率，
但稳定性从 100% 降到 70%。
```

---

### 9.5 失败分析

失败的 run：

```text
run_2 failed at step 9
run_5 failed at step 9
run_7 failed at step 9
```

#### run_2：holding 残留导致卡死

日志中出现：

```text
Bob's gripper is holding pink_polygon
```

但 legal actions 仍然给 Bob 推荐：

```text
PICK blue_square PLACE panel3
```

parser 报错：

```text
Robot Bob is already holding an object, can't pick another one.
```

根因：

```text
Sort 的 get_legal_actions() 没有考虑机器人当前 holding 状态。
```

也就是说，系统默认 `PICK obj PLACE panel` 一定在同一轮完成释放；但实际执行后可能出现机器人仍 holding 的残留状态。一旦进入这种状态，旧逻辑仍然生成 PICK 动作，导致反复失败。

#### run_5：Bob 在 panel3 抓 yellow_trapezoid IK failed

失败动作：

```text
Bob PICK yellow_trapezoid PLACE panel5
```

环境反馈：

```text
IK failed: on Bob (-0.17, 0.85, 0.28)
```

根因：

```text
符号层 reach range 可达，但实际 IK 不可解。
```

#### run_7：Chad 在 panel5 抓 yellow_trapezoid IK failed

失败动作：

```text
Chad PICK yellow_trapezoid PLACE panel6
```

环境反馈：

```text
IK failed: on Chad (0.57, 0.37, 0.21)
```

根因同样是：

```text
legal action 只做了 reach range 检查，没有做 IK 预检查或姿态放松。
```

---

## 10. 针对失败的继续修复

### 10.1 增加 Sort PLACE-only recovery 动作

为了处理 run_2 的 holding 残留，给 Sort 动作空间增加 recovery action：

```text
PLACE <object name> <location>
```

新的动作语义：

```text
如果机器人已经 holding cube，不能继续 PICK；
应该使用 PLACE-only action 把手里的 cube 释放到合法目标 panel。
```

修改位置：

```text
code/rocobench/envs/task_sort.py
SORTING_ACTION_SPACE
get_allowed_action_names()
get_legal_actions()
verify_plan_semantics()
get_task_feedback()
```

现在 Sort 支持：

```text
PICK <cube> PLACE <panel>
PLACE <cube> <panel>
WAIT
```

当检测到某机器人 holding cube 时：

```python
held_cube = self._agent_holding_cube(obs, agent_name)
```

`get_legal_actions()` 不再为该机器人生成 PICK，而是生成 PLACE-only recovery：

```text
PLACE held_cube target_panel
```

例如 run_2 的失败状态，现在推荐计划会变成：

```text
EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION PLACE pink_polygon panel4
NAME Chad ACTION WAIT
```

轻量检查中该动作通过 verifier。

---

### 10.2 Sort parser 中放松 PICK 姿态

为了降低 run_5/run_7 中的 IK failed，修改了：

```text
code/prompting/parser.py
parse_pick_action()
```

对 Sort 任务加入特殊处理：

```python
if 'SortOneBlockTask' in str(self.env):
    pick_quat = robot_state.ee_xquat.copy()
```

原因：

```text
Sort cube 随机 yaw 或 handoff 后姿态可能变化；
如果强制 gripper 匹配 cube_top site quaternion，容易造成 IK failure；
保持当前 end-effector orientation 更接近 top-down pick，通常更稳定。
```

这不是改变目标位置，只是放松抓取姿态约束。

---

### 10.3 增加轻量 IK precheck

在 Sort 中增加：

```python
can_agent_pick_cube_ik(obs, agent_name, cube_name)
```

用于在 `get_legal_actions()` 里过滤明显 IK 不可行的 PICK 候选：

```python
if not self.can_agent_pick_cube_ik(obs, agent_name, cube_name):
    continue
```

该 precheck 使用和 parser 一致的放松姿态：

```python
pick_quat = robot_state.ee_xquat.copy()
```

并使用较轻的 IK 参数，避免 prompt 构造过慢：

```python
max_resets=5
max_steps=120
allow_err=3e-2
```

目的：

```text
提前过滤符号合法但 IK 明显不可解的动作，
减少进入 parser/RRT 后才失败的概率。
```

---

### 10.4 允许 handoff panel 同 panel reposition

run_5 中 `yellow_trapezoid` 在 panel3，但 Bob 从当前位置抓取 IK failed。旧的 route 逻辑如果发现目标 panel 等于当前 panel，会直接跳过动作：

```python
if panel_name == current_panel:
    continue
```

这使得 Alice 无法把 `yellow_trapezoid` 在 panel3 上重新摆到更适合 Bob 接的位置。

因此加入：

```python
_allow_same_panel_reposition(obs, agent_name, cube_name, panel_name)
```

允许在特定 handoff panel 上做“语义不变但物理位置修正”的动作：

```text
Alice PICK yellow_trapezoid PLACE panel3
```

该动作语义上仍在 panel3，但会把物体放到 `get_target_pos()` 定义的共享 handoff 点，从而改善后续 Bob 的 IK。

目前只对：

```text
panel3 / panel5
```

这类共享 handoff panel 开放。

---

## 11. 当前代码状态与待复测内容

当前已经完成的改动：

```text
1. Sort 支持 PLACE-only recovery；
2. Sort legal actions 考虑 holding 状态；
3. Sort parser 对 PICK 姿态使用当前 EE quaternion；
4. Sort legal actions 加入轻量 IK precheck；
5. Sort 允许 handoff panel 同 panel reposition；
6. Safe parallel 仍然保留 Alice+Chad 兼容并行策略。
```

需要用户复测：

```text
sort_safe_parallel_recovery_test
sort_safe_parallel_recovery_test_10runs
```

重点观察：

```text
- run_2 类 holding 残留是否能通过 PLACE-only recovery 恢复；
- run_5 / run_7 类 IK failure 是否减少；
- 成功率是否从 70% 回升；
- 平均步数是否仍保持在 4~5 左右，而不是退回 6+。
```

建议复测命令：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code

export ROCO_DATA_DIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data
export MPLCONFIGDIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data/.matplotlib
export MUJOCO_GL=egl
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export ROCO_LLM_MODEL=llama3.3:70b

python run_dialog.py \
  --task sort \
  --run_name sort_safe_parallel_recovery_test_10runs \
  --num_runs 10 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

---

## 12. Sort Safe Parallel Recovery 复测结果与本次结论

### 12.1 本次复测对象

在上一轮 `sort_safe_parallel_test` 中，safe parallel 版本虽然把成功样本平均步数从串行 verifier 的约 `6.1` 降到了约 `4.43`，但成功率从 `100%` 降到 `70%`。

因此继续加入了三类修复后，让用户复测：

```text
sort_safe_parallel_recovery_test
```

本轮主要验证：

```text
1. holding 残留是否能通过 PLACE-only recovery 恢复；
2. Sort PICK 姿态放松是否减少 IK failed；
3. safe parallel 是否还能保持较高效率。
```

---

### 12.2 本次具体代码改动

#### 12.2.1 Sort 支持 PLACE-only recovery

新增动作：

```text
PLACE <object name> <location>
```

用途：

```text
当机器人已经 holding cube 时，不再继续给它生成 PICK 动作，
而是生成 PLACE-only recovery，让它把手中物体释放到合法目标 panel。
```

涉及文件：

```text
code/rocobench/envs/task_sort.py
```

关键点：

```python
_agent_holding_cube(obs, agent_name)
```

如果检测到 holding cube：

```text
get_legal_actions() 只生成 PLACE held_cube target_panel
不生成 PICK 动作
```

这样可以避免之前 run_2 中的错误：

```text
Bob already holding pink_polygon
但系统继续推荐 Bob PICK blue_square PLACE panel3
```

#### 12.2.2 Sort parser 放松 PICK 姿态

涉及文件：

```text
code/prompting/parser.py
```

在 `parse_pick_action()` 中对 Sort 增加特殊处理：

```python
if 'SortOneBlockTask' in str(self.env):
    pick_quat = robot_state.ee_xquat.copy()
```

目的：

```text
不再强制 gripper 匹配 cube_top site quaternion，
减少 Bob/Chad 在 panel3/panel5 handoff 处的 IK failed。
```

#### 12.2.3 Sort legal action 加入轻量 IK precheck

涉及文件：

```text
code/rocobench/envs/task_sort.py
```

新增：

```python
can_agent_pick_cube_ik(obs, agent_name, cube_name)
```

在 `get_legal_actions()` 中过滤明显 IK 不可行的 PICK 候选：

```python
if not self.can_agent_pick_cube_ik(obs, agent_name, cube_name):
    continue
```

使用较轻参数：

```text
max_resets=5
max_steps=120
allow_err=3e-2
```

目的是在 prompt/verification 阶段就过滤一部分符号合法但物理不可行的动作。

#### 12.2.4 handoff panel 同 panel reposition

为了修复 cube 已经在 handoff panel 但位置不利于下一个机器人 IK 的情况，加入：

```python
_allow_same_panel_reposition(obs, agent_name, cube_name, panel_name)
```

允许在 `panel3` / `panel5` 上执行语义不变但物理位置调整的动作，例如：

```text
Alice PICK yellow_trapezoid PLACE panel3
```

这样可以把物体重新放到 `get_target_pos()` 定义的共享 handoff 点，而不是卡在一个 Bob/Chad 很难抓的位置。

---

### 12.3 本次实验结果

对比三组实验：

```text
sort_verifier_test
sort_safe_parallel_test
sort_safe_parallel_recovery_test
```

统计结果：

```text
sort_verifier_test:
- 10 runs
- 10 成功 / 0 失败
- 成功率 100%
- 成功步数: [5, 6, 9, 6, 5, 7, 5, 7, 6, 5]
- 平均成功步数: 6.1

sort_safe_parallel_test:
- 10 runs
- 7 成功 / 3 失败
- 成功率 70%
- 成功步数: [4, 5, 5, 4, 4, 5, 4]
- 平均成功步数: 4.43
- 失败: run_2, run_5, run_7

sort_safe_parallel_recovery_test:
- 10 runs
- 9 成功 / 1 失败
- 成功率 90%
- 成功步数: [5, 5, 5, 5, 5, 5, 5, 5, 5]
- 平均成功步数: 5.0
- 失败: run_2
```

结论：

```text
recovery + IK 放松 + IK precheck 后，成功率从 70% 提升到 90%；
平均成功步数从 4.43 变为 5.0；
相比串行 verifier 的 6.1 步仍然更快。
```

也就是说，本次修复在稳定性和效率之间取得了更好的折中：

```text
串行 verifier: 稳定最高，但慢；
safe parallel: 快，但失败多；
safe parallel recovery: 成功率明显恢复，同时仍比串行快。
```

---

### 12.4 本次失败 run_2 的新问题

`sort_safe_parallel_recovery_test/run_2` 仍然失败，失败步数为：

```text
steps9_success_False
```

这次 run_2 和上一轮 run_2 不完全一样。上一轮主要是：

```text
Bob holding pink_polygon，但仍然被推荐 PICK 新物体。
```

本轮中，PLACE-only recovery 已经能生成并通过 verifier，例如在类似 holding 状态下会推荐：

```text
EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION PLACE pink_polygon panel4
NAME Chad ACTION WAIT
```

说明 holding 残留的语义层问题已经被修复。

但是本次失败 run_2 后半段出现新现象：

```text
blue_square 已完成；
yellow_trapezoid 已完成；
pink_polygon 仍显示在 panel3；
Bob 多次执行 PICK pink_polygon PLACE panel4；
feedback 显示 None；
但下一轮状态仍然是 pink_polygon on panel3。
```

典型后期状态：

```text
[Structured Task State]
- blue_square: current=panel2, target=panel2, status=done
- pink_polygon: current=panel3, target=panel4, status=not_done
- yellow_trapezoid: current=panel6, target=panel6, status=done

[Recommended Plan]
NAME Alice ACTION WAIT
NAME Bob ACTION PICK pink_polygon PLACE panel4
NAME Chad ACTION WAIT
```

该动作多次反馈为：

```text
FB: None
```

但环境状态没有推进到 done。

这说明当前失败不再是 verifier/semantic 层面的问题，而更像是执行层或状态判定层问题：

```text
1. feedback_manager 的目标姿态检查通过；
2. RRT/IK 也没有在反馈阶段报错；
3. 但真实执行后，物体没有被成功移动到 panel4，或者没有被判定为在 panel4。
```

可能原因包括：

```text
- Bob 的 pick-place 实际没有稳定抓住 pink_polygon；
- place 后物体掉回或仍接近 panel3；
- panel4 目标点/释放高度/return_home 导致物体未进入 align_threshold；
- get_reward_done() / get_cube_panel() 的判定和动作目标反馈之间存在差异；
- parser feedback 只检查末端目标，不检查执行后物体最终状态。
```

---

### 12.5 本次改进的有效点

本次改进确实解决或缓解了之前的两类问题：

#### holding 残留问题明显缓解

之前 run_2 会在 Bob holding 时继续推荐 PICK，导致 parser 报：

```text
Robot Bob is already holding an object, can't pick another one.
```

现在 legal action 已能根据 holding 状态生成 PLACE-only recovery，不再把 holding robot 当成空手 robot。

#### IK failed 问题明显减少

上一轮失败中：

```text
run_5: Bob IK failed
run_7: Chad IK failed
```

本轮 10 次中没有看到这两个 run 继续失败，整体成功率从 70% 回升到 90%，说明：

```text
Sort PICK 姿态放松 + 轻量 IK precheck 是有效的。
```

---

### 12.6 当前剩余问题与下一步方向

当前剩余最大问题是：

```text
动作反馈通过，但执行后物体状态没有变化，导致同一动作被重复执行到步数耗尽。
```

这说明 verification layer 还缺少“执行后状态推进检查”。

下一步建议：

#### 1. 增加 action effect monitor

每轮执行后检查：

```text
如果执行的是 PICK obj PLACE target，
那么下一轮 obj 应该更接近 target，或者 status 应该变成 done。
```

如果连续两次执行同一动作但 object panel 不变，则触发特殊 recovery：

```text
- 换另一个可达机器人；
- 换目标偏移点；
- 换成 PLACE-only / reposition；
- 或将该动作临时标记为 execution ineffective，而不是继续重复。
```

#### 2. 对 panel4 final placement 做目标点/释放高度检查

本次唯一失败集中在：

```text
Bob PICK pink_polygon PLACE panel4
```

需要重点检查：

```text
panel4 target pose 是否足够稳定；
Bob place 后是否 return_home 过早导致物体被拖走；
pink_polygon 是否因为形状/朝向导致释放失败；
get_cube_panel() 是否把接近 panel4 的物体仍判成 panel3。
```

#### 3. 为 final placement 增加 post-place tolerance 或 retry target

如果物体放到 panel4 附近但未进入判定阈值，可以尝试：

```text
panel4_middle 的多个候选位置；
稍高/稍低的释放高度；
Bob 当前 EE quaternion 的不同姿态；
执行同一动作前先 reposition 到 panel4 更中心的位置。
```

#### 4. 保留当前 safe parallel recovery 方案

从实验结果看，当前版本相比纯串行更快，相比上一版并行更稳：

```text
100% / 6.1 步  -> 串行 verifier
70% / 4.43 步 -> 原 safe parallel
90% / 5.0 步  -> safe parallel recovery
```

因此不建议回退到纯串行。下一步应该继续针对 `Bob PICK pink_polygon PLACE panel4` 的执行效果做局部修复。

---

## 13. 针对 recovery_test 剩余失败的继续修复

### 13.1 剩余失败现象

`sort_safe_parallel_recovery_test` 提升到了：

```text
9 / 10 成功
平均成功步数 5.0
```

唯一失败仍是 `run_2`。后期状态显示：

```text
blue_square 已在 panel2 完成；
yellow_trapezoid 已在 panel6 完成；
pink_polygon 仍在 panel3；
Bob 多次执行 PICK pink_polygon PLACE panel4；
feedback 显示 None；
但下一轮 pink_polygon 仍被描述为 on panel3。
```

这说明该问题不是普通语义错误，也不是 pre-RRT 的明显 IK/碰撞错误，而是：

```text
动作被认为可执行，但执行后 object state 没有产生有效推进。
```

### 13.2 新判断

在同一状态下，legal actions 中其实同时存在：

```text
Alice: PICK pink_polygon PLACE panel3
Bob: PICK pink_polygon PLACE panel4
```

其中 Alice 的动作是 handoff panel 的 same-panel reposition：

```text
语义上仍然把 pink_polygon 放在 panel3，
但物理上会把它放到 get_target_pos(panel3) 定义的共享 handoff 点。
```

之前推荐计划更偏向 Bob 的 final placement，因为 final target 分数更高。结果如果 Bob 的执行对物体没有实际效果，就会反复尝试同一动作直到步数耗尽。

因此判断：

```text
当 cube 已经 nominally 在 panel3/panel5，
但不在共享 handoff target 附近时，
应该优先做 same-panel reposition，
再让接收机器人做 final placement。
```

### 13.3 本次代码修复

修改文件：

```text
code/rocobench/envs/task_sort.py
```

修改位置：

```python
get_recommended_plan() -> action_score()
```

新增优先级：

```python
if (
    target_panel == current_panel
    and self._allow_same_panel_reposition(obs, agent_name, cube_name, target_panel)
):
    return 150
```

含义：

```text
如果一个动作是 panel3/panel5 上的 same-panel reposition，
并且当前 cube 距离共享 handoff target 仍超过阈值，
则给它比普通 final placement 更高的推荐分数。
```

这样在类似失败状态下，推荐计划会更倾向于：

```text
Alice PICK pink_polygon PLACE panel3
```

而不是继续重复：

```text
Bob PICK pink_polygon PLACE panel4
```

预期动作流变为：

```text
1. Alice reposition pink_polygon on panel3 shared handoff point
2. Bob PICK pink_polygon PLACE panel4
```

虽然多一步，但能避免无效重复。

### 13.4 轻量检查

已做编译检查：

```bash
python -m py_compile rocobench/envs/task_sort.py prompting/parser.py prompting/plan_prompter.py
```

结果：通过。

### 13.5 待复测

建议继续复测：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code

export ROCO_DATA_DIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data
export MPLCONFIGDIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data/.matplotlib
export MUJOCO_GL=egl
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export ROCO_LLM_MODEL=llama3.3:70b

python run_dialog.py \
  --task sort \
  --run_name sort_safe_parallel_reposition_test_10runs \
  --num_runs 10 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

重点观察：

```text
- 原 recovery_test/run_2 类 pink_polygon on panel3 卡住问题是否消失；
- 成功率是否从 90% 继续接近 100%；
- 平均步数是否仍在 5 左右；
- 是否出现过度 reposition 导致步数增加的问题。
```
