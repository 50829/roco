# Rope 任务 plan 模式优化记录

## 1. 背景与目标

`rope` / `MoveRopeTask` 的目标是：两个机器人 Alice 和 Bob 分别抓住长绳两端，将绳子越过障碍墙，最终把绳子两端放入窄槽中。

代码入口：

```text
code/run_dialog.py
code/rocobench/envs/task_rope.py
code/prompting/plan_prompter.py
code/prompting/parser.py
code/prompting/feedback.py
```

运行时，`run_dialog.py` 会对 rope 做特殊设置：

```python
args.output_mode = 'action_and_path'
args.split_parsed_plans = True
args.control_freq = 20
args.max_failed_waypoints = 0
```

这意味着 rope 不只要求大模型输出高层动作，还必须输出每个机器人 gripper 的四个三维路径点：

```text
NAME Alice ACTION PICK rope_front_end PATH [(x,y,z), ...]
NAME Bob ACTION PICK rope_back_end PATH [(x,y,z), ...]
```

本轮优化目标是让 plan 模式下的 rope 尽量稳定遵循：

```text
先两端都 PICK
再两端一起 PUT
```

而不是让某个机器人单独移动、单独释放绳子，导致 rope 动力学状态变坏。

---

## 2. 初始问题：模型输出了不存在的 LIFT 动作

### 现象

早期日志中，大模型在 Alice/Bob 已经抓住绳子后，经常输出：

```text
EXECUTE
NAME Alice ACTION LIFT rope_front_end PATH [...]
NAME Bob ACTION LIFT rope_back_end PATH [...]
```

随后 parser 报错：

```text
Parsing failed! Action LIFT rope_front_end PATH [...] can't be parsed.
```

### 原因

`task_rope.py` 的动作空间实际只支持：

```text
PICK <obj> PATH <path>
PUT <obj> <location> PATH <path>
WAIT
```

parser 中没有 `LIFT` 动作。

但 rope prompt 里有自然语言：

```text
must lift up the rope together
lift rope up before moving it to PUT
```

模型把自然语言里的 `lift` 误当成动作名，导致反复输出非法动作。

此外，原 prompt 里还有一个不一致示例：

```text
ACTION PLACE rope_front_end groove_left_end PATH ...
```

但 rope 的合法放置动作应是：

```text
PUT rope_front_end groove_left_end PATH ...
```

### 修改

文件：

```text
code/rocobench/envs/task_rope.py
```

在 `ROPE_ACTION_SPACE` 中加入明确约束：

```text
Valid ACTION names are only PICK, PUT, and WAIT.
Never output LIFT, MOVE, PLACE, LOWER, RAISE, or DROP.
To lift or move the rope while holding it, use PUT <obj> <location> PATH <path> with high intermediate PATH coordinates; do not use a LIFT action.
```

并将误导性示例从：

```text
ACTION PLACE rope_front_end groove_left_end PATH ...
```

改为：

```text
ACTION PUT rope_front_end groove_left_end PATH ...
```

### 效果

修改后，模型不再主要卡在 `LIFT` 解析失败上，开始输出合法动作名：

```text
PICK
PUT
WAIT
```

但随后暴露出新的问题：IK 不收敛、WAIT 被任务反馈误判、单端操作导致 rope 状态变坏。

---

## 3. 第二个问题：通用 plan prompt 混入 Sort 专用内容

### 现象

rope 的 system prompt 末尾出现了与 rope 无关的内容：

```text
Object target panels are fixed. Do not swap object goals.
For Sort Cubes specifically, the fixed goals are: blue_square -> panel2, ...
```

这会污染 rope 任务的规划，让模型看到 panel / cube / handoff 之类与 rope 无关的信息。

### 原因

`code/prompting/plan_prompter.py::get_plan_prompt()` 中写死了 Sort Cubes 规则。

### 修改

将 `get_plan_prompt(env)` 改成优先调用任务自己的 `central_plan_prompt()`：

```python
def get_plan_prompt(env: MujocoSimEnv):
    if hasattr(env, "central_plan_prompt"):
        try:
            task_prompt = env.central_plan_prompt()
        except TypeError:
            task_prompt = env.central_plan_prompt([])
        if task_prompt:
            return task_prompt

    return """
Reason about the task step-by-step...
"""
```

同时把 Sort 专用内容放回 `task_sort.py::SORT_TASK_PLAN_PROMPT`。

### 效果

rope prompt 不再混入 Sort 内容，每个任务使用自己的 plan prompt：

```text
sort      -> task_sort.py 的 SORT_TASK_PLAN_PROMPT
rope      -> task_rope.py 的 ROPE_TASK_PLAN_PROMPT
sandwich  -> task_sandwich.py 的 SANDWICH_PLAN_PROMPT
pack      -> task_pack.py 的 PACK_PLAN_PROMPT
```

这一步解决的是 prompt 污染问题，不直接解决 IK。

---

## 4. 第三个问题：WAIT 在 prompt 中允许，但 task feedback 中禁止

### 现象

为了解决某个机器人 IK 失败，我们让失败机器人 WAIT，例如：

```text
NAME Alice ACTION PICK rope_front_end PATH [...]
NAME Bob ACTION WAIT PATH [...]
```

但是 feedback 返回：

```text
Task Constraints:
 faild, Bob's ACTION is not supported
```

### 原因

`task_rope.py::get_task_feedback()` 原逻辑为：

```python
if 'PLACE' in action_str or 'WAIT' in action_str or 'MOVE' in action_str:
    task_feedback += f"{agent_name}'s ACTION is not supported"
```

也就是说 prompt 里允许 WAIT，但任务反馈层又禁止 WAIT，自相矛盾。

### 修改

将 `WAIT` 从 unsupported 列表中移除，只禁止真正非法的动作：

```python
if 'PLACE' in action_str or 'MOVE' in action_str or 'LIFT' in action_str:
    task_feedback += f"{agent_name}'s ACTION is not supported"
```

并在 prompt 中明确：

```text
WAIT PATH <path>: do nothing. Because this task uses action_and_path mode, even WAIT must include exactly four coordinates; use four copies of the current gripper location.
```

### 效果

WAIT 不再被 task feedback 判为非法，可以作为保守 fallback 动作使用。

---

## 5. 第四个问题：rope PICK 目标高度太低，导致桌面碰撞和 IK 失败

### 现象

日志中出现：

```text
Goal Step Alice (-1.25, 0.46, 0.26); Bob (-0.55, 0.31, 0.26):
  - Collision detected: collided object pairs: Alice-table
```

还有：

```text
IK failed: on Bob (-0.55, 0.31, 0.52)
```

更极端的状态中，rope_front_end 掉到：

```text
rope_front_end: (-0.83, 0.93, 0.05)
```

模型 PATH 中最后点可能写了 z=0.49 / 0.52，但 parser 最终 PICK goal 不是直接用 LLM PATH 最后点，而是调用：

```python
task_rope.py::get_target_pos(agent_name, rope_end)
```

原逻辑只加：

```python
ret[2] += 0.1
```

因此 rope z=0.16 时，PICK goal z 约为 0.26；rope z=0.05 时，PICK goal z 约为 0.15，或后续修改后为 0.25，仍然太低。

### 修改过程

#### 第一次修改

先将 rope PICK 目标高度从：

```python
ret[2] += 0.1
```

提高到：

```python
ret[2] += 0.2
```

目的是让 gripper 不要贴桌面抓 rope，减少 `Alice-table` 碰撞。

#### 第二次修改

发现 rope 掉得太低时，`+0.2` 仍不够；但如果 rope 当前 z 比较高，简单加 0.35 又可能超过 prompt 中的高度上限 `0.55`。

因此改成上下界裁剪：

```python
ret[2] = np.clip(ret[2] + 0.35, 0.42, 0.54)
```

当前逻辑为：

```text
PICK 目标 z = clamp(rope 当前 z + 0.35, 最低 0.42, 最高 0.54)
```

示例：

| rope 当前 z | PICK 目标 z |
|---:|---:|
| 0.05 | 0.42 |
| 0.16 | 0.51 |
| 0.25 | 0.54 |
| 0.36 | 0.54 |

### 效果

- 避免了 rope 端点很低时 goal z 仍然贴桌面；
- 避免了 rope 当前较高时目标 z 超过 `0.55`；
- 对桌面碰撞有帮助，但不能完全解决 x/y 太偏导致的 IK 不收敛。

---

## 6. 第五个问题：单端操作 rope 会把 rope 拖坏

### 现象

一次失败轨迹中，历史为：

```text
Round#0:
Alice: PICK rope_front_end
Bob: WAIT

Round#1:
Alice: PUT rope_front_end groove_left_end
Bob: PICK rope_back_end
```

随后状态变成：

```text
rope_front_end: (-0.83, 0.93, 0.05)
rope_back_end: (-0.84, 0.67, 0.36)
Alice's gripper: (0.13, -0.06, 0.50), holding nothing
Bob's gripper: (-0.82, 0.66, 0.36), holding rope_back_end
```

此时模型继续让 Alice 抓 rope_front_end：

```text
NAME Alice ACTION PICK rope_front_end PATH [..., (-0.83, 0.93, 0.49)]
```

但 Alice 对该位置 IK 失败：

```text
IK failed: on Alice (-0.83, 0.93, 0.25)
IK failed: on Alice (-0.83, 0.93, 0.49)
```

### 原因

rope 是长柔性物体。单端抓取、单端放置会导致另一端被拖拽、掉落或跑到机器人难以到达的位置。

因此不能把 rope 当成普通刚体物品处理。

### 设计决策

明确 rope 的阶段顺序：

```text
阶段 1：Alice 和 Bob 必须先同时/分别完成两端 PICK，使两端都被抓住。
阶段 2：只有当两端都被抓住后，才允许 Alice 和 Bob 同一轮双 PUT。
```

更具体：

```text
如果两人都没拿 rope：
  Alice PICK rope_front_end
  Bob PICK rope_back_end

如果只有 Alice holding：
  Alice WAIT
  Bob PICK rope_back_end

如果只有 Bob holding：
  Alice PICK rope_front_end
  Bob WAIT

如果两人都 holding：
  Alice PUT rope_front_end groove_left_end
  Bob PUT rope_back_end groove_right_end
```

---

## 7. 第六个修改：加入 Recommended Rope Stage Plan

### 修改内容

在 `task_rope.py` 中新增：

```python
_agent_holding_rope(obs, agent_name)
_agent_ee_pos(obs, agent_name)
_format_path(pts)
_wait_action(obs, agent_name)
_pick_action(obs, agent_name, rope_end)
_put_action(obs, agent_name, rope_end, groove_end)
get_recommended_plan(obs)
format_legal_actions_prompt(obs)
```

其中 `get_recommended_plan(obs)` 是确定性阶段策略：

```python
if Alice empty and Bob empty:
    Alice PICK rope_front_end
    Bob PICK rope_back_end
elif Alice holding and Bob empty:
    Alice WAIT
    Bob PICK rope_back_end
elif Alice empty and Bob holding:
    Alice PICK rope_front_end
    Bob WAIT
else:
    Alice PUT held_end groove_left_end
    Bob PUT held_end groove_right_end
```

`format_legal_actions_prompt(obs)` 会把当前阶段和推荐计划插入 prompt：

```text
[Rope Stage State]
- Alice holding: nothing / rope_front_end / rope_back_end
- Bob holding: nothing / rope_front_end / rope_back_end

[Recommended Rope Stage Plan]
Follow this staged plan exactly unless environment feedback says it failed:
EXECUTE
NAME Alice ACTION ...
NAME Bob ACTION ...
```

### 为什么这样改

rope 的难点不在于“知道要把东西放进槽”，而在于动作阶段不能乱。

让 LLM 自由规划时，它可能会为了规避某个 IK 失败而选择单端 PUT 或混合 PICK+PUT，但这会让 rope 状态更差。

因此用 deterministic recommended plan 给模型强引导，减少不稳定行为。

---

## 8. 第七个修改：feedback 层硬约束阶段顺序

只加 prompt 还不够，所以在 `get_task_feedback()` 中加入硬约束。

当前约束包括：

### 8.1 禁止非法动作

```python
if 'PLACE' in action_str or 'MOVE' in action_str or 'LIFT' in action_str:
    task_feedback += f"{agent_name}'s ACTION is not supported"
```

### 8.2 禁止 PICK 和 PUT 混在同一轮

```python
if has_put and has_pick:
    task_feedback += "Do not mix PICK and PUT in the same round..."
```

### 8.3 禁止单机器人 PUT

```python
if has_put and not all('PUT' in action for action in actions.values()):
    task_feedback += "The rope must be PUT by both robots at the same time..."
```

### 8.4 如果无人 holding，必须 Alice/Bob 双 PICK

```python
if num_holding == 0:
    Alice must PICK rope_front_end
    Bob must PICK rope_back_end
```

### 8.5 如果只有一个机器人 holding

要求：

```text
holding robot: WAIT
empty robot: PICK other rope end
```

### 8.6 如果两人都 holding

要求：

```text
both robots should PUT in same round
```

---

## 9. 当前效果与已知问题

### 已改善的问题

1. `LIFT` 解析失败减少；
2. Sort prompt 不再污染 rope；
3. `WAIT` 不再被 rope task feedback 判为 unsupported；
4. rope PICK goal 高度不再贴桌面，也不超过 0.55；
5. prompt 和 task feedback 都强制“先两端 PICK，再两端 PUT”；
6. 当前已有历史中，`data/rope_prompt_fix/run_0` 出现过成功：

```text
data/rope_prompt_fix/run_0/steps1_success_True.json
{"step": 1, "success": true, "timed_out": false, "elapsed_time": 85.17...}
```

该成功轨迹大致为：

```text
step_0:
Alice PICK rope_front_end
Bob PICK rope_back_end

step_1:
Alice PUT rope_front_end groove_left_end
Bob PUT rope_back_end groove_right_end
```

### 仍存在的问题

#### 问题 1：某些初始状态下 Bob PICK IK 失败

典型失败：

```text
IK failed: on Bob (-0.55, 0.31, 0.52)
```

原因是 Bob 当前 gripper 和 rope_back_end 的位置关系不一定好，且 Panda 的末端姿态约束可能过硬。

#### 问题 2：某些坏状态下 Alice 重抓 rope_front_end IK 失败

典型失败：

```text
rope_front_end: (-0.83, 0.93, 0.05)
IK failed: on Alice (-0.83, 0.93, 0.49)
```

这种状态通常是之前单端操作造成的。阶段约束加入后，应该减少这类状态，但如果已经进入坏状态，仍然难救。

#### 问题 3：parser 对 rope PICK 使用固定四元数

`parser.py::parse_pick_action()` 中 rope 特判为：

```python
if 'Rope' in str(self.env):
    site_name = obj_name
    pick_pos = self.env.get_target_pos(agent_name, site_name)
    pick_quat = np.array([1,0,0,0])
```

这会强迫机器人末端达到固定姿态 `[1,0,0,0]`。对于 rope，真正关键是位置接近 rope end 并激活 weld，固定姿态可能让 IK 更难。

建议下一步改为：

```python
pick_quat = robot_state.ee_xquat.copy()
```

也就是保持当前 gripper 姿态，降低 IK 难度。

---

## 10. 下一步计划

### P0：继续验证当前阶段约束

运行：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-09/26220478/code

python run_dialog.py \
  --task rope \
  --run_name rope_stage_pick_then_put \
  --num_runs 1 \
  --skip_display \
  --comm_mode plan \
  --llm_source llama3.3:70b \
  --run_timeout 600
```

重点检查：

1. 是否仍出现 `LIFT`；
2. 是否仍出现 `PLACE`；
3. 是否还出现 `PICK + PUT` 同轮混合；
4. 是否还出现单机器人 PUT；
5. 是否遵循：

```text
双 PICK -> 双 PUT
```

### P1：如果仍有 IK failed，修改 rope PICK 姿态约束

文件：

```text
code/prompting/parser.py
```

建议修改：

```python
if 'Rope' in str(self.env):
    site_name = obj_name
    pick_pos = self.env.get_target_pos(agent_name, site_name)
    pick_quat = robot_state.ee_xquat.copy()
```

替代当前：

```python
pick_quat = np.array([1,0,0,0])
```

### P2：如果仍不稳定，做 rope 专用 deterministic fallback

可以在 prompter fallback 中加入 rope 专用策略：

```text
状态 A：两人 empty -> 直接返回 env.get_recommended_plan()
状态 B：单人 holding -> holding WAIT + empty PICK
状态 C：双人 holding -> 双 PUT
```

并在 LLM 多次 replan 失败后自动尝试，而不是完全依赖模型继续生成路径。

### P3：按 seed 统计成功率

至少跑：

```bash
python run_dialog.py --task rope --run_name rope_stage_5runs --num_runs 5 --skip_display --comm_mode plan --llm_source llama3.3:70b --run_timeout 600
```

记录每个 run 的失败原因：

| run | 成功 | 失败阶段 | 主要原因 |
|---|---|---|---|
| 0 | ? | PICK / PUT | IK / collision / timeout |

---

## 11. 当前涉及文件汇总

已修改：

```text
code/prompting/plan_prompter.py
code/rocobench/envs/task_sort.py
code/rocobench/envs/task_rope.py
```

本 rope 优化主要集中于：

```text
code/rocobench/envs/task_rope.py
```

建议下一步可能修改：

```text
code/prompting/parser.py
```

用于降低 rope PICK 的姿态 IK 难度。
