# 个人实训报告：Sort Cubes 任务优化

## 1. 个人负责内容

本次实训中，我主要负责 RocoBench 多机器人协同操作中的 **Sort Cubes** 任务优化。该任务要求 Alice、Bob、Chad 三个机器人协作，将不同形状/颜色的方块移动到对应目标 panel：

- `blue_square -> panel2`
- `pink_polygon -> panel4`
- `yellow_trapezoid -> panel6`

由于三个机器人可达范围不同，跨区域移动必须通过共享 handoff panel 完成：Alice 与 Bob 通过 `panel3` 交接，Bob 与 Chad 通过 `panel5` 交接。因此我的工作重点不是简单调 prompt，而是让大模型输出的计划既能被解析，又符合机器人可达性、IK、碰撞和交接约束。

## 2. 初始问题分析

实验初期，Sort 任务主要存在以下问题：

1. **LLM 输出格式正确但物理上不可执行**  
   例如模型会让 Bob 抓取实际够不到的物体，导致 `Reachability failed` 或 `IK failed`。

2. **缺少方向性 handoff 约束**  
   方块可能被放到后续机器人无法接手的位置，虽然动作表面合法，但会造成任务死局。

3. **并行动作带来碰撞风险**  
   三个机器人同时动作时，Bob/Chad 等机器人可能在共享工作区发生碰撞。

4. **失败处理过粗**  
   早期如果一个机器人的动作失败，会把整轮 response 中所有动作都加入 forbidden，误伤其他本来可以执行的机器人动作。

5. **执行中残留 holding 状态**  
   机器人可能已经抓住 cube，但下一轮仍被推荐执行 PICK，导致 “already holding an object” 的错误。

## 3. 我的主要改进工作

### 3.1 为 Sort 任务加入合法动作枚举

我在 `code/rocobench/envs/task_sort.py` 中加入了 Sort 专用 planning helper，包括：

- `get_plan_state_prompt(obs)`：把当前 cube 位置、目标 panel、完成状态、可达机器人整理进 prompt；
- `get_legal_actions(obs)`：根据当前状态为每个机器人生成合法动作；
- `get_recommended_plan(obs)`：基于合法动作生成推荐计划；
- `format_legal_actions_prompt(obs)`：把合法动作列表和推荐计划写入 prompt。

这样模型不再完全自由生成动作，而是被限制在当前状态下可执行的动作集合中。

### 3.2 加入方向性 handoff 规则

根据机器人可达范围，我为不同 cube 设计了方向性转运规则。例如：

- Alice/Bob 通过 `panel3` 交接；
- Bob/Chad 通过 `panel5` 交接；
- 不允许把 cube 放到后续无人能处理的位置。

这减少了“当前动作可执行，但后续任务进入死局”的情况。

### 3.3 在 plan 层加入前置校验和 fallback

我在 `code/prompting/plan_prompter.py` 中加入了对 LLM 输出的校验逻辑：

- 必须包含 `EXECUTE`；
- 每个机器人必须 exactly one action；
- action 必须来自 legal actions；
- 同一轮不能多个机器人操作同一个 cube 或放到同一目标；
- 失败动作会加入 forbidden，避免模型重复犯错。

同时加入 deterministic partial fallback：当 LLM 多次 replan 仍失败时，系统会尝试推荐计划或推荐计划的子集，优先执行可行的单机器人/双机器人动作，避免因为某一个机器人失败导致整轮停滞。

### 3.4 精确处理失败机器人

我修改了失败动作 ban 的策略。以前一轮动作中只要 Bob 失败，Alice 和 Chad 的动作也可能被一起禁掉。现在会从反馈中识别失败机器人，例如 `IK failed: on Bob`，只 ban Bob 当前失败动作，保留 Alice/Chad 的可执行动作。

这个改动提高了任务恢复能力。

### 3.5 修复 handoff 点和 IK 问题

实验发现原来的 `panel3` / `panel5` 放置点虽然放置方能放，但接收方后续抓取时容易 IK failed。因此我调整了 Sort 任务中的 handoff target position，使共享 panel 的放置点更适合“放的人能放，接的人也能抓”。

此外，我还在 parser 中对 Sort 的 PICK 姿态做了放松处理：不强制匹配 cube top site 的 quaternion，而使用当前 end-effector 姿态，从而减少因为物体 yaw 或 handoff 后姿态变化导致的 IK 失败。

### 3.6 增加 recovery 与 safe parallel

为了解决机器人已经 holding cube 的情况，我加入了 `PLACE <object> <panel>` 形式的 PLACE-only recovery 动作。如果机器人手中已有 cube，系统不再推荐 PICK，而是推荐先把手中物体放到合适位置。

在效率方面，我尝试了 safe parallel 策略：

- 每轮最多 2 个 active actions；
- Bob 作为中间机器人不与其他机器人并行；
- 只允许 Alice 和 Chad 在 footprint 不重叠时并行；
- 避免多个机器人操作同一个 cube 或同一个目标 panel。

## 4. 实验结果

我记录并比较了多个版本的实验结果：

| 实验版本 | 结果 | 说明 |
| --- | --- | --- |
| `sort_legal_plan_test` | 4/7 成功 | 加入合法动作和初步 fallback 后，仍有碰撞失败 |
| `sort_verifier_test` | 10/10 成功 | 串行 verifier 版本稳定性最高，平均约 6.1 step |
| `sort_safe_parallel_test` | 7/10 成功 | safe parallel 提高效率，成功样本平均约 4.43 step，但稳定性下降 |
| `sort_safe_parallel_recovery_test` | 9/10 成功 | 加入 PLACE-only recovery 和 IK 改进后，稳定性恢复到 90% |

从结果看，单纯追求并行会提高效率但降低稳定性；加入 recovery、IK precheck、姿态放松和更精确的失败处理后，可以在效率和成功率之间取得更好的平衡。

## 5. 遇到的困难与解决方法

1. **自然语言 prompt 不足以保证物理可执行**  
   解决：加入程序化 legal action 枚举和 plan verifier。

2. **符号合法不代表 IK 可解**  
   解决：加入轻量 IK precheck，并对 Sort PICK 姿态做任务特化放松。

3. **并行执行容易碰撞**  
   解决：限制并行数量，引入 footprint 检查，只允许低风险并行。

4. **失败恢复困难**  
   解决：失败动作只 ban 对应机器人，并加入 partial fallback 和 PLACE-only recovery。

## 6. 个人收获

通过 Sort Cubes 任务，我认识到多机器人具身操作不是简单让大模型“生成计划”，还需要把任务规则、机器人可达性、运动规划约束和失败恢复机制结合起来。大模型适合做高层规划，但在实际系统中必须加入确定性校验、可执行动作空间和反馈重规划机制。

这次实训让我对以下方面有了更深入理解：

- 多机器人协同任务中的 handoff 设计；
- LLM plan 与底层 parser/RRT/IK 的接口关系；
- 如何根据实验日志定位失败原因；
- 如何在成功率和执行效率之间做工程权衡；
- 如何把失败案例转化为规则、校验和 fallback 机制。

## 7. 后续改进方向

后续如果继续优化 Sort 任务，可以考虑：

1. 用更系统的 symbolic planner / BFS 生成 Sort 的最短可行动作序列；
2. 对不同 seed 的失败进行更细粒度分类；
3. 将 IK precheck 和碰撞风险估计做得更准确；
4. 保留串行稳定策略作为兜底，在 safe parallel 失败时自动降级。
