# Sort 任务 plan 模式改进记录

## 背景

目标是让 RocoBench 的 `sort` 任务在 `comm_mode=plan` 下稳定跑通。任务规则是 3 个机器人把 3 个物体放到固定目标 panel：

- `blue_square -> panel2`，主要由 Alice 完成。
- `pink_polygon -> panel4`，主要由 Bob 完成。
- `yellow_trapezoid -> panel6`，主要由 Chad 完成。

机器人可达范围固定：

- Alice: `panel1, panel2, panel3`
- Bob: `panel3, panel4, panel5`
- Chad: `panel5, panel6, panel7`

因此跨机器人协作必须通过共享 handoff panel 完成：

- Alice/Bob 之间用 `panel3`
- Bob/Chad 之间用 `panel5`

## 最初遇到的问题

### 1. LLM 会生成格式正确但物理上非法的动作

典型失败：

```text
NAME Alice ACTION WAIT
NAME Bob ACTION PICK yellow_trapezoid PLACE panel5
NAME Chad ACTION WAIT
```

当 `yellow_trapezoid` 在 `panel3` 时，Bob 从逻辑上看靠近 panel3，但实际环境反馈是：

```text
Reachability failed: Out of reach: Bob
```

LLM 收到反馈后仍反复输出同一个错误动作，导致 step 反复失败。

### 2. 物体会被搬到无人能接的位置

之前的自由规划允许 Alice 把 `yellow_trapezoid` 放到 `panel3`，但之后 Bob/Chad 可能抓不到，任务进入死局。

根因是 prompt 只描述了“可达 panel”，没有把 sort 任务中的方向性 handoff 强约束给模型。

### 3. 高层合法动作不等于 IK 可行

加入 legal action 后，又出现了另一类失败：

```text
NAME Alice ACTION PICK blue_square PLACE panel2
NAME Bob ACTION PICK pink_polygon PLACE panel4
NAME Chad ACTION PICK yellow_trapezoid PLACE panel6
```

环境反馈：

```text
IK failed: on Bob (-0.44, 0.64, 0.18)
```

这说明动作在“panel 可达”层面合法，但 Bob 实际抓取位姿 IK 不可解。

### 4. 失败动作 ban 过粗

早期处理方式是：只要一个 plan 失败，就把该 response 中所有非 WAIT 动作都加入 forbidden。

问题是：

- Bob 的动作 IK failed。
- Alice 和 Chad 的动作可能是可执行的。
- 但旧逻辑把 Alice/Chad 也一起禁掉。

结果 fallback 没法利用“部分可执行动作”继续推进任务。

### 5. 过度保守的 panel 占用规则会卡死初始状态

我曾尝试加入规则：不要把 cube 放到已经有未完成 cube 的 panel 上。

后来轻量检查发现，这个规则过度保守。例如初始状态可能是：

```text
blue_square is on panel5
pink_polygon is on panel1
yellow_trapezoid is on panel3
```

如果禁止使用已占用 handoff panel，合法动作会退化成全 WAIT，任务直接卡死。

所以这个尝试被撤销。

## 解决思路

核心原则：不要只靠 LLM 自己理解任务规则，而是在 plan 模式外层加确定性约束和兜底。

具体分三层：

1. **生成合法动作集合**：当前状态下每个机器人只能从枚举出的动作里选。
2. **LLM 输出后先校验**：不合法就不交给 parser/RRT。
3. **LLM 多次失败后 deterministic fallback**：自动尝试推荐计划的子集，避免一个机器人失败拖垮整轮。

## 修改点一：`task_sort.py` 中加入 sort 专用 planning helper

文件：

```text
code/rocobench/envs/task_sort.py
```

新增了这些方法：

### `get_plan_state_prompt(obs)`

生成结构化状态，告诉 planner：

- 每个 cube 当前在哪个 panel。
- 目标 panel 是什么。
- 是否已经完成。
- 哪些机器人当前能 reach。

### `is_cube_done(obs, cube_name)`

判断 cube 是否已经在目标 panel，避免再次移动已完成物体。

### `can_agent_reach_cube(obs, agent_name, cube_name)`

根据实际物体 top site 检查机器人是否能抓到 cube。

### `can_agent_place_panel(agent_name, panel_name)`

根据 `get_target_pos()` 检查机器人是否能把物体放到目标 panel。

### `_sort_route_targets(obs, agent_name, cube_name)`

加入 sort 任务的方向性约束。物体只能沿正确方向移动：

- `blue_square` 最终去 `panel2`：从右往左时通过 `panel5 -> panel3 -> panel2`。
- `pink_polygon` 最终去 `panel4`：Alice/Chad 只能把它交给 Bob。
- `yellow_trapezoid` 最终去 `panel6`：Alice/Bob/Chad 只能向右交接。

这样可以防止 LLM 把物体搬到错误方向，或者搬到后续无人能处理的位置。

### `get_legal_actions(obs)`

为每个机器人生成当前合法动作列表。LLM 后续必须从这里选动作。

### `get_recommended_plan(obs)`

基于合法动作给一个 greedy 推荐计划，用于 prompt 强提示和 fallback。

策略上最多并行动作数限制为 2，避免三个机器人同时动导致 IK/碰撞风险变大。

### `format_legal_actions_prompt(obs)`

把合法动作和推荐计划写进 system prompt：

```text
[Legal Actions]
Alice:
- WAIT
- PICK pink_polygon PLACE panel3 (recommended)
...

[Recommended Plan]
EXECUTE
NAME Alice ACTION ...
...
```

## 修改点二：修正 handoff panel 的放置点

文件：

```text
code/rocobench/envs/task_sort.py
```

函数：

```python
get_target_pos(agent_name, target_name)
```

原逻辑对 `panel3` / `panel5` 使用“根据放置机器人变化”的偏移点。

实际日志和 pkl 检查显示：

- Alice 放 `panel3` 的旧位置大约是 `(-0.28, 0.60, 0.5)`。
- Bob 后续从这个位置抓 `pink_polygon` 会 IK failed。
- Bob 放 `panel3` 的位置大约是 `(-0.52, 0.40, 0.5)`，更适合作为 Alice/Bob 的共享 handoff 点。
- Bob 放 `panel5` 的位置大约是 `(0.52, 0.40, 0.5)`，Chad 后续可以接。

所以改成固定共享 handoff 点：

```python
if target_name == 'panel3':
    ret[0] -= 0.12
    ret[1] -= 0.1

if target_name == 'panel5':
    ret[0] += 0.12
    ret[1] -= 0.1
```

这样 handoff 点满足更关键的条件：**放的人能放，接的人也能抓**。

## 修改点三：`plan_prompter.py` 中加入 legal action 校验

文件：

```text
code/prompting/plan_prompter.py
```

新增了动作解析和校验逻辑：

- `_extract_action_lines(response)`
- `_validate_against_legal_actions(response, legal_actions, forbidden_actions)`

校验内容包括：

1. 必须包含 `EXECUTE`。
2. 必须每个机器人 exactly one action。
3. action 必须在 `[Legal Actions]` 列表里。
4. 同一轮不能多个机器人 pick 同一个 object。
5. 同一轮不能多个机器人 place 到同一个 target。
6. 不能重复本轮已经失败的 forbidden action。

这样 LLM 即使输出格式看似正确，只要动作不在合法集合里，就不会进入 parser/RRT。

## 修改点四：失败动作只 ban 失败机器人

新增方法：

- `_extract_agents_from_text(text)`
- `_ban_actions_from_response(response, forbidden_actions, agents=None)`
- `_format_forbidden_actions(forbidden_actions)`

环境反馈里如果出现：

```text
IK failed: on Bob
Out of reach: Bob
Illegal action for Bob
```

现在只 ban Bob 的当前失败动作，不会误伤 Alice/Chad。

这点很关键。之前 Bob 失败时，Alice/Chad 的可执行动作也被禁掉，导致 fallback 不能推进。

## 修改点五：加入 deterministic partial fallback

新增方法：

- `_get_recommended_response(obs)`
- `_response_from_actions(actions)`
- `_try_parse_and_feedback(obs, response)`
- `_partial_fallback_responses(obs, legal_actions, forbidden_actions)`

流程是：

1. LLM 正常 replan。
2. 如果多次 replan 都失败，进入 deterministic fallback。
3. 先尝试 env 推荐计划。
4. 如果推荐计划整体失败，尝试推荐计划的子集：
   - 两个机器人动作组合
   - 单机器人动作
5. 如果推荐计划子集也被 forbidden 或 IK 卡住，再枚举其它合法的一机器人/两机器人动作组合。

这解决了典型问题：Bob 的动作 IK failed，但 Alice/Chad 的动作可执行。现在系统不会整轮放弃，而是会执行可行子集继续推进环境状态。

## 修改点六：减少历史 prompt 污染

原来 `compose_round_history()` 会把较多 LLM 历史 response 放回 prompt，模型容易复制旧错误动作。

现在只保留最近几轮真正执行过的 `[Executed Action]`，减少失败样本污染。

## 修改点七：修复反馈重复 append

早期分支里有些 feedback 会被 append 两次，导致 prompt 里重复出现同一条错误信息。

已去掉重复 append，避免让模型在 replan 时被冗余错误反馈干扰。

## 中间失败尝试与原因

### 尝试 1：只靠 prompt 强调固定目标和 handoff

结果：失败。

原因：LLM 仍会重复输出 reach/IK 不可行的动作，例如 Bob 抓不到的 `yellow_trapezoid`。

结论：自然语言约束不够，必须加程序化合法动作枚举。

### 尝试 2：加入 legal actions

结果：部分改善，但仍失败。

原因：legal action 只保证高层 reach/panel 逻辑合法，不保证 IK 一定可解。

典型失败是 Bob 对 `pink_polygon` 的 IK failed。

结论：需要结合环境反馈，失败后禁止具体失败动作，并尝试部分动作 fallback。

### 尝试 3：失败后 ban 整个 response 的非 WAIT 动作

结果：失败。

原因：Bob 失败时把 Alice/Chad 的可行动作也禁了，导致系统无法执行部分可行计划。

结论：必须从反馈文本中识别失败机器人，只 ban 对应机器人的动作。

### 尝试 4：禁止把 cube 放到已占用 panel

结果：失败。

原因：sort 的 handoff panel 天然可能临时承载多个待处理物体。禁止占用会导致初始状态没有合法动作，只剩全 WAIT。

结论：这个规则过度保守，已撤销。

### 尝试 5：使用 `xvfb-run` 本地跑完整评测

结果：当前环境里 GL 初始化失败：

```text
gladLoadGL error
```

后来发现 `MUJOCO_GL=egl` 可以启动轻量环境检查，因此推荐 GPU 上使用 EGL 跑。

## 最终解决结果

用户在 GPU 上复测 4 次，均已跑通。

这说明当前组合有效：

1. sort 任务专用 legal action 枚举。
2. 固定共享 handoff 点。
3. LLM 输出前置校验。
4. 精确 ban 失败机器人动作。
5. deterministic partial fallback。

## 当前推荐运行方式

在 GPU 环境中运行：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code

export ROCO_DATA_DIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data
export ROCO_EVAL_OUTPUT_DIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data
export MPLCONFIGDIR=/inspire/qb-ilm2/project/26summer-camp-09/26220478/data/.matplotlib
export MUJOCO_GL=egl
export OLLAMA_BASE_URL=http://127.0.0.1:11434

python run_dialog.py \
  --task sort \
  --run_name sort_plan_test \
  --num_runs 1 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

多次评测可以改：

```bash
python run_dialog.py \
  --task sort \
  --run_name sort_plan_4runs \
  --num_runs 4 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

## 产物位置

输出目录在：

```text
/inspire/qb-ilm2/project/26summer-camp-09/26220478/data/<run_name>/run_<id>/
```

典型产物：

```text
step_0/
  env_init.pkl
  env_end.pkl
  llm_plan_0.pkl
  rrt_plan_0.pkl
  actions_0.pkl
  execute.mp4
  prompts/
    replan0_*.json
    replan0_feedback_*.json
    fallback_*.json   # 只有触发 fallback 时出现

stepsN_success_True.json
stepsN_success_True.html
```

注意：只有真正执行了动作的 step 才会有 `execute.mp4`。如果某个 step 只是 LLM/replan 失败，没有执行物理动作，就不会有视频。

## 需要注意的点

这次修改中，`task_sort.py` 不只是增加 prompt helper，还修改了 `get_target_pos()` 中 panel3/panel5 的 handoff 放置点。这个改动会影响 sort 任务中放置目标的实际坐标。

这样做的原因是原 handoff 点本身导致接收方 IK failed；如果只在 prompt 层规避，仍可能反复失败。

如果后续要求严格“不改环境任务定义”，可以把 legal action 和 fallback 逻辑保留在 `prompting` 层，但 handoff 点问题仍需要通过某种方式解决，例如：

- 在 parser 层为 sort handoff 特判目标点；或
- 在 prompt/action candidate 中使用显式坐标目标；或
- 保留当前 `get_target_pos()` 修正并在报告中说明这是 handoff 可达性修复，不是更改目标任务成功条件。

## 补充复盘：`sort_legal_plan_test/run_1` 到 `run_6`

这批日志位于：

```text
data/sort_legal_plan_test/run_1
...
data/sort_legal_plan_test/run_6
```

结果汇总：

| run | 结果 | 结束 step | 说明 |
| --- | --- | --- | --- |
| run_1 | success=True | step 4 | 成功 |
| run_2 | success=True | step 3 | 成功 |
| run_3 | success=False | step 9 | Bob/Chad 并行动作碰撞后卡住 |
| run_4 | success=False | step 9 | Bob/Chad 并行动作碰撞后卡住 |
| run_5 | success=True | step 4 | 成功 |
| run_6 | success=True | step 3 | 成功 |

### run_3 / run_4 的失败根因

这两个失败不是 reachability，也不是 IK，而是并行动作碰撞。

典型失败计划：

```text
EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION PICK blue_square PLACE panel3
NAME Chad ACTION PICK yellow_trapezoid PLACE panel6
```

环境反馈：

```text
Collision detected: collided object pairs: Chad-Bob
```

当状态类似下面这样时：

```text
blue_square is on panel5
pink_polygon is in panel4
yellow_trapezoid is on panel5
```

Bob 和 Chad 都要从右侧共享区域附近取物/放物，两条机械臂同时进入相邻工作区，RRT/碰撞检查判定 `Chad-Bob` 碰撞。

### 为什么当时没有自动恢复

旧逻辑的问题是：

- 一旦某个 multi-robot plan 失败，就会把该 response 里的非 WAIT 动作加入 forbidden。
- 对于 collision，这个处理是错误的。
- 因为 collision 通常表示“两个动作同时做不安全”，不表示 Bob 单独做不安全，也不表示 Chad 单独做不安全。

结果是：

1. Bob 的 `PICK blue_square PLACE panel3` 被禁。
2. Chad 的 `PICK yellow_trapezoid PLACE panel6` 也被禁。
3. 后续 LLM 只能反复输出 forbidden action 或 all WAIT。
4. 任务从 step_3/step_4 开始一直空转到 step_9 失败。

### 二次修正

为这个问题又做了两处修改。

#### 1. sort 推荐计划默认改成单机器人动作

文件：

```text
code/rocobench/envs/task_sort.py
```

修改：

```python
max_parallel_actions = 1
```

原因：sort 任务足够短，即使顺序执行也能在 10 step 内完成。相比并行，单机器人推荐计划更稳，能避免 Bob/Chad 同时进入 shared workspace 导致碰撞。

注意：这只是推荐计划更保守，不是完全禁止并行。Legal actions 仍然允许 LLM 提出并行动作，但默认 fallback/recommended plan 会倾向串行执行。

#### 2. collision 反馈不再 ban 单个动作

文件：

```text
code/prompting/plan_prompter.py
```

新增逻辑：

```python
def _should_ban_individual_actions(self, feedback: str) -> bool:
    return "Collision detected" not in feedback
```

并给 collision feedback 增加更明确提示：

```text
Collision feedback means the concurrent robot combination is unsafe.
Try fewer simultaneous actions, preferably one active robot and others WAIT.
```

含义：

- 如果是 reachability / IK / illegal action，说明某个具体机器人动作本身可能不可行，可以 ban 对应动作。
- 如果是 collision，说明并发组合不安全，不应该 ban 单个动作。
- 这样 fallback 可以继续尝试：

```text
Bob only
Chad only
```

而不是把 Bob/Chad 都禁掉。

### 修正后的预期行为

以 run_3 的卡住状态为例，旧推荐计划是：

```text
Bob: PICK blue_square PLACE panel3
Chad: PICK yellow_trapezoid PLACE panel6
```

现在推荐计划会变成单动作优先，例如：

```text
Alice: WAIT
Bob: WAIT
Chad: PICK yellow_trapezoid PLACE panel6
```

下一步再让 Bob 处理 blue_square：

```text
Alice: WAIT
Bob: PICK blue_square PLACE panel3
Chad: WAIT
```

最后 Alice 把 blue_square 放到 panel2。

这样牺牲一点并行速度，但明显提高稳定性。
