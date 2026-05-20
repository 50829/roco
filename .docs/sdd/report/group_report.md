# 多机器人协同具身操作实验报告：任务级调试与仲裁机制实现

## 1 实验目标与整体思路

本实验基于 `RocoBench` 多机器人操作环境，目标是在现有代码框架 `roco-main` 上接入大语言模型，使多个机器人能够在有限可达性、动作顺序限制、碰撞约束和路径规划约束下完成协作任务。与直接追求一个“通用大一统方案”不同，我们采用的是更偏工程化的自下而上方法：

1. 先把六个 task 分给组员分别调试，逐个复现失败现象。
2. 结合代码里的动作空间、任务提示词和环境反馈，分析每个 task 为什么失败。
3. 按 task 做细粒度修复，包括 prompt 改写、规则补充、模式切换和 verifier 增强。
4. 当各 task 的基础成功率提升到一定水平后，再引入仲裁机制，对候选计划做统一校验。
---

## 2 代码框架与运行流程

### 2.1 主要代码入口

本次实验主要围绕以下几类文件展开：

| 文件/目录 | 作用 | 本实验关注点 |
|---|---|---|
| `run_dialog.py` | 总控入口，负责加载环境、创建 parser / prompter / planner，并逐轮执行 | 实验运行流程、Plan/Dialog 模式切换、超时控制 |
| `evaluator.py` | 统一跑六个任务并统计结果 | 成功率统计、默认超时、默认 mode |
| `prompting/plan_prompter.py` | 集中式规划与 verifier 逻辑 | `Pack` 专用 verifier，后续仲裁机制的代码参考 |
| `prompting/dialog_prompter.py` | 多智能体对话式协作逻辑 | Dialog 模式的轮次组织与反馈注入 |
| `rocobench/envs/task_*.py` | 各任务环境定义、动作空间、任务提示词和 task-specific 规则 | 每个 task 的失败原因与修复依据 |

结合 `run_dialog.py` 的实现，框架主要支持两类协作方式：

- `plan` / `chat`：使用 `SingleThreadPrompter`，由统一规划器生成本轮动作。
- 其他对话模式：使用 `DialogPrompter`，让机器人先讨论再汇总动作。

`evaluator.py` 里对应的默认评测任务共有 6 个：`sort`、`cabinet`、`rope`、`sweep`、`sandwich`、`pack`，每个任务默认运行 5 次，单次超时 600 秒。

### 2.2 任务与机器人配置

| 任务 | 环境文件 | 机器人配置 | 关键约束 |
|---|---|---|---|
| Sort Cubes | `task_sort.py` | Alice / Bob / Chad | 面板可达性、接力交接、固定目标面板 |
| Arrange Cabinet | `task_cabinet.py` | Alice / Bob / Chad | 先开门再取物、保持门开启、放置姿态 |
| Move Rope | `task_rope.py` | Alice / Bob | 双臂同时抓绳、IK 可达性、越墙高路径 |
| Sweep Floor | `task_sweep.py` | Alice / Bob | 先对齐再 SWEEP、Alice 必须 WAIT |
| Make Sandwich | `task_sandwich.py` | Chad / Dave | 严格配方顺序、每轮仅一个 PUT |
| Pack Grocery | `task_pack.py` | Alice / Bob | 空槽位、路径分离、高空走廊、防碰撞 |

### 2.3 单轮运行流程

单轮实验的运行逻辑可以概括为：

```python
load_task_env(task_name)
init_parser_and_feedback_manager()
init_prompter(plan_or_dialog_mode)

for step in range(max_runner_steps):
    obs = env.get_obs()
    candidate_plan = prompter.generate(obs, history, feedback)
    parsed_plan = parser.parse(candidate_plan)
    env_feedback = feedback_manager.check(parsed_plan, obs)
    if env_feedback indicates invalid:
        update_feedback_and_replan()
    else:
        execute_plan_in_env()
        save_step_result()
```

从中可以看出，失败并不一定是 LLM 本身的问题，也可能出在解析失败、动作不合法、IK 不可解、并行动作冲突等环节。所以后面的修复工作不能只改 prompt，还得结合 task 规则和执行反馈一起处理。

---

## 3 各任务问题分析与修复

### 3.1 baseline 阶段的模式选择

在 baseline 阶段，我们先不做 task-specific 修改，直接测试 `Plan` 和 `Dialog` 两种模式的表现，目的是判断每个任务更适合哪种协作方式。

| 模式 | Sort Cubes | Arrange Cabinet | Move Rope | Sweep Floor | Make Sandwich | Pack Grocery |
|---|---:|---:|---:|---:|---:|---:|
| Plan | 0/5  | 2/5  | 1/5  | 0/5  | 1/5  | 0/5  |
| Dialog | 1/5  | 1/5  | 0/5  | 2/5  | 0/5  | 0/5  |

从实验现象和项目论文实验数据来看，`Sort` 和 `Sweep` 这类更依赖局部可达性和显式配合的任务，更适合在对话或强反馈模式下处理；`Cabinet`、`Sandwich`、`Pack` 更偏流程化或规则化，适合用 `Plan` 做统一约束；`Rope` 虽然可以用 `Plan`，但必须加入很强的路径和 IK 反馈规则。

### 3.2 Sort Cubes：可达性误判与错误交接

`task_sort.py` 中给出了非常明确的结构化约束：每个机器人只能到达固定面板区间，且每个立方体都有固定目标面板。同时，代码中还提供了 `get_legal_actions()`、`verify_plan_semantics()` 和 `_sort_route_targets()` 这套更严格的有向交接逻辑。

- `reachable_panels` 明确限制了 Alice / Bob / Chad 的可达区域。
- `_sort_route_targets()` 指定了交接只能通过 `panel3` 和 `panel5` 这两个共享中转面板。
- `verify_plan_semantics()` 会拒绝“不在 legal actions 里”的动作，也会拒绝移动已经完成的 cube。

实验中最常见的问题是，LLM 会把上一轮失败动作当成已经成功执行，于是继续让不具备可达性的机器人去抓取目标，或者把 cube 移到一个后续无人可接的面板上。说到底，这里不是“不会分类”，而是当前状态理解错了，同时又给了模型过大的自由搬运空间。

我们的修复做法有两步：

1. 在 prompt 中强调“失败动作不会改变环境状态”。
2. 将自由规划改成“只允许从 legal actions 中选择”，并采用定向 handoff 策略。

```python
for agent in agents:
    legal_actions = get_legal_actions(obs, agent)
    if previous_plan_failed:
        remove_failed_action_from_candidates()
    choose_only_from(legal_actions)

if direct_goal_unreachable:
    use_handoff_panel(panel3_or_panel5)
```

![Sort Cubes 失败案例截图](./images/image-1.png)

### 3.3 Arrange Cabinet：动作顺序正确但失败会被重复继承

`task_cabinet.py` 的动作空间写得很清楚：先 `PICK <handle>`，再 `OPEN <handle>`，开门后还要 `WAIT` 保持门开着，之后才能把 `cup` 或 `mug` 放到正确的 coaster 上。这个 task 的难点并不在语义理解，而在于动作链很长，且任意一步失败都会连锁影响后续步骤。

实际调试中，我们遇到两类问题：

- 前一步的放置没有成功，但后续 planner 仍然认为杯子已经放到位了。
- 放置目标姿态偏差较大，导致杯子掉落或者后续抓取 IK 变差。

代码里也能看到开发者已经在 `coaster_pos` 上做过人工修正，例如对 `z` 和 `x` 做了偏移。这也说明放置位姿本身就是影响 task 成功率的重要因素。

针对 `Cabinet`，我们最后主要做了三类修复：

1. 保留跨 step 的失败反馈，避免“同一动作无限重复”。
2. 对放置目标位姿做保守修正，提高放置稳定性。
3. 在 prompt 中强化“开门阶段”和“取放阶段”之间的依赖。

```python
if handle_not_open:
    only_allow(["PICK handle", "OPEN handle", "WAIT hold_door"])
elif object_not_on_coaster:
    plan_pick_and_place(object, correct_coaster)

if previous_place_failed:
    keep_feedback_for_next_step()
    lower_or_adjust_place_pose()
```

![Arrange Cabinet 失败链条示意图](./images/image-2.png)
### 3.4 Sweep Floor：没有对齐就 SWEEP

`task_sweep.py` 是最典型的“高层动作看起来简单，实际上前提非常严格”的任务。代码里的 `SWEEP_ACTION_SPACE` 已经把约束写得很细：

- `SWEEP` 只能由 Bob 执行。
- Bob `SWEEP` 时 Alice 必须 `WAIT`。
- 只有当 Alice 距离目标小于 `0.20` 且 Bob 距离目标小于 `0.45` 时，才允许 `SWEEP`。
- 如果没有对齐，必须让两人都 `MOVE` 到同一个 cube。

一开始失败的原因很简单：LLM 被“任务目标是清扫 cube”这个高层语义带偏，倾向于过早输出 `SWEEP`，但实际环境要求是先完成双机器人对齐。

我们的修复不是简单提醒“先 MOVE”，而是把对齐规则直接写进动作空间和 prompt，同时把这类 task 更倾向放到可讨论的模式里，让两个机器人先确认目标一致。

```python
if not aligned_same_cube(alice_dist < 0.20, bob_dist < 0.45):
    action[Alice] = MOVE(target_cube)
    action[Bob] = MOVE(target_cube)
else:
    action[Alice] = WAIT
    action[Bob] = SWEEP(target_cube)
```

![Sweep Floor 错误 SWEEP 示例](./images/image-3.png)

### 3.5 Move Rope：IK 失败后需要换抓取端和路径

`task_rope.py` 的规则说明非常具体：两只机械臂必须同时抓住绳索两端，路径要越过障碍墙，而且如果某个机器人对某一端 IK 失败，就不能死磕同一个目标，而应该动态交换 rope end 的分配。代码里已经直接写了：

- “Do not hard-code which robot picks which rope end”
- “If one robot fails IK for a rope end, that robot should try the other rope end”
- 越墙时路径高度建议 `z = 0.52 ~ 0.54`

这个 task 的问题不在于 LLM 不知道目标，而在于高层规划很容易写出“语义上说得通、几何上却走不通”的路径。尤其是路径压得过低时，模型就会反复在墙前失败。

我们的修复分两部分：

1. 抓取阶段按 reachability 动态分配 rope end，而不是固定 Alice 前端、Bob 后端。
2. 放置阶段使用高抬弧线，并在失败后切换目标端或 waypoint 模板。

```python
assign_end_by_reachability()
if ik_fail(agent, rope_end):
    swap_rope_end_assignment()

for waypoint in path:
    enforce_high_arc_near_wall(z >= 0.52)

if put_fail_again:
    replace_with_alternate_put_template()
```

![Move Rope 目标交换策略示意图](./images/image-4.png)

### 3.6 Make Sandwich：配方顺序约束没有被持续遵守

`task_sandwich.py` 把配方顺序写得非常明确：

- `SANDWICH_RECIPES` 定义了各类 sandwich 的合法顺序。
- `PICK` 只能拿当前 recipe 中“下一个正确食材”。
- 每轮只能有一个机器人执行 `PUT`。

但是 baseline 阶段的 LLM 经常出现的问题是：开始几步看起来合理，走到中后期后忽略 recipe 顺序，持续尝试把错误食材放到错误层上，随后陷入重复。

所以这个 task 的核心不是“动作语法”，而是“长期维持顺序一致性”。我们的修复方案包括：

1. 在 prompt 中重复强调“只允许 PICK 正确的下一个食材”。
2. 对反复失败动作加入 blacklist。
3. 在 plan 后面增加 verifier，专门检查顺序是否合法。

```python
next_item = recipe[next_recipe_index]

if action is PICK and obj != next_item:
    reject_plan()

if action_repeatedly_failed:
    blacklist.add(action)

if verifier_detects_wrong_order(candidate_plan):
    rewrite_plan_to_follow_recipe()
```

![Make Sandwich 顺序约束错误案例](./images/image-5.png)

### 3.7 Pack Grocery：抓取阶段和放置阶段都可能碰撞

`task_pack.py` 和 `plan_prompter.py` 中的 `Pack` 相关规则是整个项目里写得最细的。除了动作空间本身规定了 `PICK` / `PLACE` / `PATH`，`plan_prompter.py` 里还提供了 `compose_pack_verifier_system_prompt()` 和 `verify_pack_plan()`，把 `Pack` 的二阶段 verifier 实现成了一个独立校验器。

这个任务的失败主要分成两个阶段：

第一阶段，抓取时两只机械臂会同时向桌面中部靠近，导致路径空间拥挤；

第二阶段，即使成功抓取，如果同时把物体放入相邻 slot，下降时仍然会碰撞。`Pack verifier` 里专门写了：

- 不允许把物体放进已占用 slot；
- 同时 PLACE 时，两目标 slot 距离应尽量大于 `0.35`；
- Alice 和 Bob 需要走不同的高空 corridor；
- Bob 不能用低空、中央 waypoint。

我们对这个 task 的修复也是分两轮完成的：

1. 第一轮先解决“抓取目标太近”问题，让两臂优先选更分散的物体。
2. 第二轮再解决“放置走廊太近”问题，引入 slot 距离约束和高空分离路径。

```python
if both_robots_pick_from_center():
    reassign_farther_objects()

if both_robots_place_same_round():
    choose_distant_empty_slots(min_xy_dist=0.35)
    alice_path = high_left_front_corridor()
    bob_path = high_back_right_corridor()

if verifier_rejects_plan:
    rewrite_candidate_plan()
```

![Pack Grocery 初始失败案例](./images/image-6.png)

![Pack Grocery 第一轮修正效果](./images/image-7.png)

![Pack Grocery 第二轮修正效果一](./images/image-8.png)

![Pack Grocery 第二轮修正效果二](./images/image-9.png)

---

## 4 仲裁机制设计与实现

### 4.1 为什么在 task 微调之后还要加仲裁

前面逐 task 调试下来以后，我们发现很多错误虽然出现在不同任务里，但表现形式其实很接近：

- 输出格式不稳定，解析器拿不到合法动作。
- 候选动作没有真正满足 task 规则。
- 失败动作被重复使用。
- 同一轮动作组合在局部看似正确，合并后却互相冲突。

也就是说，只靠每个 task 单独改 prompt，最后还是会留下不少框架层面的漏洞。所以在局部修复之外，我们又补了一层独立的后验校验，也就是仲裁机制。

### 4.2 代码落点与实现思路

我们的做法是：

1. 先让 planner 生成一个 candidate plan。
2. 再把 candidate plan 连同历史反馈、黑名单和 task 规则送给 verifier。
3. verifier 判断候选计划是否满足规则。
4. 如果不满足，直接输出修正后的 plan。

### 4.3 仲裁机制的输入与输出

仲裁模块的输入包括：

- 当前 step 的结构化观测。
- planner 或 dialog 产生的候选计划。
- 上一轮或多轮的环境反馈。
- 已知失败动作黑名单。
- 当前 task 的动作格式和约束规则。

输出则只有一个：最终交给 parser 和执行器的可执行计划。

### 4.4 仲裁伪代码

```python
def arbitration_agent(obs, candidate_plan, env_feedback, blacklist, task_rules):
    if not valid_format(candidate_plan):
        candidate_plan = rewrite_format(candidate_plan)

    actions = parse(candidate_plan)
    if not actions:
        return safe_fallback_plan(task_rules)

    for action in actions:
        if action in blacklist:
            action = replace_with_safe_action(action, task_rules, obs)

    if not satisfy_task_rules(actions, obs, task_rules):
        actions = repair_by_rules(actions, obs, env_feedback, task_rules)

    if repeated_failure_detected(actions, env_feedback):
        actions = switch_strategy(actions, task_rules, obs)

    return format_as_execute(actions)
```

### 4.5 这套机制具体解决了什么

从实验过程来看，仲裁机制最直接解决的是三类问题：

1. 把“格式正确但语义错误”的计划拦在执行前。
2. 把“上一轮已经失败过的动作”从候选计划里剔掉。
3. 当 planner 仍然过于乐观时，给出更保守但可执行的修正版动作。

所以，仲裁机制并不是拿来替代 LLM 做完整规划的，更像是整个系统里的最后一道保险。
---

## 5 实验设置

### 5.1 评测指标

本实验只统计成功率。统计方式与 `evaluator.py` 保持一致：读取每个 run 结果 JSON 中的 `success` 字段，计算

$$
\text{Success Rate}=\frac{\text{success count}}{\text{total count}}
$$

这样做的原因很简单：本实验更关心“任务最终有没有完成”，而不是在尚未稳定跑通时过早比较步数和细粒度失败分布。

### 5.2 运行配置

参考 `evaluator.py`：

- 测试任务：`sort`、`cabinet`、`rope`、`sweep`、`sandwich`、`pack`
- 每个任务运行次数：5 次
- 单次超时：600 秒
- 运行入口：`run_dialog.py`
- 统计入口：`evaluator.py`

### 5.3 对比方法

本实验分三步做对比：

1. baseline：直接测试 `Plan` 和 `Dialog`，先找出每个 task 更适合哪种模式。
2. task 级微调：分别微调各 task 后，对比 `llama3.3` 和 `qwen3.5`。
3. 仲裁机制：在更稳定的 `llama3.3` 上继续加入仲裁机制，观察是否还能进一步提升成功率。

---

## 6 实验结果

### 6.1 baseline：不同模式的初始表现

| 模式 | Sort Cubes | Arrange Cabinet | Move Rope | Sweep Floor | Make Sandwich | Pack Grocery |
|---|---:|---:|---:|---:|---:|---:|
| Plan | 0/5  | 1/5 | 1/5  | 0/5  | 0/5 | 0/5  |
| Dialog | 1/5  | 1/5  | 0/5  | 0/5 | 2/5 | 0/5  |

从 baseline 结果来看，两种协作模式在不同任务上的适配差异已经比较明显。`Sort Cubes` 和 `Sweep Floor` 在 `Dialog` 下略好于 `Plan`，说明这两类任务更依赖局部可达性判断和机器人之间的显式协商；尤其是 `Sweep Floor`，如果没有先对齐再执行动作，集中式规划很容易直接输出不合时宜的 `SWEEP`。相比之下，`Arrange Cabinet`、`Move Rope` 和 `Make Sandwich` 在 `Plan` 下至少能够部分跑通，说明这几类任务更适合先用统一规划把动作顺序、目标分配和整体流程固定下来。`Pack Grocery` 在两种模式下都是 `0/5`，说明它的主要瓶颈并不只是选错协作模式，而是放置阶段的路径分离、槽位选择和碰撞约束还没有被表达清楚。这一组结果也为后续 task 级微调提供了方向：`Sort` 和 `Sweep` 更需要补足局部规则与协商约束，`Cabinet`、`Rope`、`Sandwich` 则更适合在现有 `Plan` 基础上继续强化规则和反馈利用。

### 6.2 task 级微调后：不同模型的对比

| 模型 | Sort Cubes | Arrange Cabinet | Move Rope | Sweep Floor | Make Sandwich | Pack Grocery |
|---|---:|---:|---:|---:|---:|---:|
| llama3.3 | 3/5  | 4/5  | 3/5  | 4/5  | 3/5  | 0/5  | 
| qwen3.5 | 2/5  | 3/5  | 1/5  | 3/5  | 4/5  | 0/5  | 

经过 task 级微调之后，两种模型的差异开始变得更清楚。`llama3.3` 在 `Sort Cubes`、`Arrange Cabinet`、`Move Rope` 和 `Sweep Floor` 上都优于 `qwen3.5`，其中 `Move Rope` 的差距最明显，说明在需要结合失败反馈调整路径和目标分配的任务里，`llama3.3` 的稳定性更好。`qwen3.5` 只在 `Make Sandwich` 上略高于 `llama3.3`，说明它在顺序性较强、动作空间相对简单的任务里也有一定优势，但整体上不如 `llama3.3` 稳定。`Pack Grocery` 两个模型都还是 `0/5`，这一点也说明该任务当前的瓶颈主要不在模型本身，而在于碰撞约束、空中走廊和后验校验还不够强。综合来看，`llama3.3` 更适合作为后续仲裁机制实验的基础模型，因为它在大多数任务上表现更均衡，能更稳定地把 task 级修复落实到实际执行中。

### 6.3 加入仲裁机制后的增益

| 设置 | Sort Cubes | Arrange Cabinet | Move Rope | Sweep Floor | Make Sandwich | Pack Grocery |
|---|---:|---:|---:|---:|---:|---:|
| llama3.3 | 3/5  | 4/5  | 3/5  | 4/5  | 3/5  | 0/5  | 
| llama3.3 + Arbitration | 4/5  | 5/5  | 5/5  | 5/5  | 5/5  | 0/5  | 
| Increase Rate | 1/5  | 1/5  | 2/5  | 1/5  | 2/5  | 0/5  |

从加入仲裁机制后的结果来看，它的作用并不是简单重复 task 级 prompt，而是在已有微调基础上进一步做统一收口。`Sort Cubes` 从 `3/5` 提升到 `4/5`，`Arrange Cabinet` 从 `4/5` 提升到 `5/5`，说明仲裁机制对格式稳定性、失败动作回避和局部规则补全确实有帮助。`Move Rope` 和 `Make Sandwich` 的提升最明显，最终都达到 `5/5`，这表明在需要处理失败反馈、动作合法性和顺序约束的任务里，后验校验层能显著减少重复错误。`Pack Grocery` 仍然保持 `0/5`，说明仅靠当前这版仲裁还不足以解决其路径级碰撞问题，这个任务后面仍然需要继续加强放置走廊约束和几何层面的修正。整体来看，仲裁机制对大多数任务都带来了额外增益，尤其适合作为 task 级修复之后的统一增强模块。

---

## 7 实验总结与经验

从 baseline 结果来看，直接调用大模型去完成多机器人协作任务，整体效果并不稳定，而且不同任务对协作模式的敏感性也很强。像 `Sort Cubes` 和 `Sweep Floor` 这类更依赖局部可达性判断和机器人之间显式配合的任务，在初始阶段更适合用 `Dialog` 来暴露“谁能做什么”；而 `Arrange Cabinet`、`Move Rope`、`Make Sandwich` 这类带有明确动作顺序或统一流程的任务，则更适合先用 `Plan` 把整体步骤固定下来。也就是说，baseline 阶段最重要的作用并不是给出多高的成功率，而是帮助我们先判断每个 task 的主要难点究竟出在协作模式、规则表达，还是执行反馈没有被正确利用。

在完成任务级微调后，实验结果说明这种“逐 task 排查和修补”的方式是有效的。相比直接依赖原始 prompt，task 级修改之后，大部分任务的成功率都有了明显改善，而且不同模型之间的差异也开始变得清楚。这里最关键的经验是，很多失败其实并不是单纯因为模型能力不够，而是因为动作前提没有写清楚、失败反馈没有进入下一轮规划，或者某些 task-specific 规则没有被显式表达出来。把这些局部问题拆开处理以后，`Sort`、`Cabinet`、`Rope`、`Sweep` 和 `Sandwich` 的表现都比 baseline 更稳定，这也说明 task 级调试是后续整体改进的基础。

在此基础上加入仲裁机制后，系统又获得了进一步提升，说明统一的后验校验层确实是有必要的。尤其是在 `Arrange Cabinet`、`Move Rope` 和 `Make Sandwich` 这类任务中，仲裁机制能够继续减少重复失败、格式错误和局部规则违例带来的损失，把 task 级修复积累下来的经验进一步统一起来。结合整套实验过程来看，这次项目比较有效的一条路线就是：先通过 baseline 找到任务和协作模式的大致对应关系，再通过 task 级微调逐步修补具体问题，最后再用仲裁机制做统一收口。最终的提升并不是来自某一个单独技巧，而是这三步逐层叠加之后形成的结果。


---

## 8 小组分工与个人贡献说明

### 8.1 【成员 A：孙鹏】

孙鹏在本次项目中主要负责 `Arrange Cabinet` 和 `Make Sandwich` 两个任务的调试与优化，同时也承担了整体 baseline 搭建和跨任务问题梳理的工作。具体来说，他先对 `Plan` 与 `Dialog` 两类提示词模板进行了拆解，结合 `task_*.py` 中的动作空间、约束规则和环境反馈，逐项分析各任务失败是来自信息缺失、约束表达不充分，还是反馈没有被后续规划正确利用。在此基础上，他整理并实现了一系列具有通用性的改进思路，包括强化失败动作不改变环境状态的提示、引入失败动作黑名单、补充跨 step 失败记忆，以及在候选计划后增加 verifier 进行二次校验。

在任务调试方面，孙鹏重点完成了 `Arrange Cabinet` 与 `Make Sandwich` 的问题定位和修复，并围绕放置失败重复继承、配方顺序约束缺失等典型现象提出了对应的 prompt 和机制改进。由于整体推进较快，他还进一步对 `Sort`、`Sweep Floor`、`Move Rope`、`Pack Grocery` 等任务进行了补充调试，希望先为小组建立一个可用的整体 baseline，再逐步分任务细化。以 `llama3.3:70b` 为基础模型进行测试后，他整理得到各任务的初步成功率表现，其中 `cabinet`、`rope` 和 `make sandwich` 均取得了较稳定的结果。这部分工作一方面为后续小组分工提供了起点，另一方面也帮助小组较早明确了“先做 task 级修复，再引入统一仲裁机制”的整体实验路线。

### 8.2 【成员 B：沈鼎丁】

沈鼎丁主要负责 baseline 评测、Sort Cubes 主线优化和 Move Rope 早期探索。实训初期，他完成环境配置、模型服务启动与评测链路验证，对六个任务进行 baseline 运行，并从日志中整理出 parser 错误、IK failed、collision、语义非法动作、timeout 等主要失败类型，为小组确定优化优先级和后续改进方向提供依据。

在 Sort Cubes 的 Plan mode 中，他针对三机器人 handoff、可达范围不同以及 LLM 容易生成不可执行动作等问题，实现了 legal actions、方向性 handoff、plan-level verifier、失败动作精确屏蔽、partial fallback 和 recovery 机制，使 Sort 从早期 4/7 成功提升到 verifier 版本 10/10 成功；在进一步尝试 safe parallel 后，又通过 recovery 机制将成功率稳定到 9/10。该任务也成为小组验证任务级 verifier / fallback 思路的主要样例之一。

在 Move Rope 中，他负责早期问题定位和阶段化约束探索，修复了非法动作、prompt 污染和 WAIT/PATH 不一致等问题，并提出“双 PICK → 双 PUT”的阶段化策略。虽然 Rope 的最终最佳效果由组员在路径与执行层继续优化取得，但他的早期工作明确了 Rope 的核心瓶颈在于 IK、PATH 和柔性物体状态变化。第二天小组融合新架构时，他参与将 Sort 中验证有效的 verifier / fallback 思路接入统一 arbitration 框架，使个人分支中的任务级经验转化为小组整体方案的一部分。

### 8.3 【成员 C：熊禹翔 】

熊禹翔同学主要负责 Pack Grocery 任务的失败分析与改进。首先分析 baseline 失败原因，将问题拆分为高层规划和路径规划两类：前者包括机器人分工、操作顺序和物体-槽位匹配不稳定，后者表现为 LLM 生成的 PATH 不合法或存在碰撞。对于路径规划问题，他设计并实施了一个路径碰撞判断实验，说明模型即使在简单线性插值场景下也不能稳定判断碰撞，因此不应直接依赖 LLM 规划路径。基于此，他引入了外部的路径规划算法，让 LLM 只输出高层动作，算法自动生成“拿起-移动-放下”的简单 PATH。对于高层规划问题，他探索了全局规划，局部规划两条技术路线。他通过实验发现模型能力无法做出精确的全局规划，基于此，他着眼于局部规划，探索了未来预测、候选动作自评和 reject action 机制，提升了架构的短期决策、失败后改出的能力，提升了 Pack 任务的成功率。