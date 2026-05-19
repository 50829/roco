# 代码架构说明

## 1. 顶层入口

### `evaluator.py`

批量评测脚本，会依次测试 6 个任务，每个任务默认 5 次 run。

重要说明：评测时该文件可能被官方替换，因此不要依赖修改它来提高最终成绩。

### `run_dialog.py`

单任务主入口，负责：

1. 根据 `--task` 创建环境；
2. 创建 LLMRunner；
3. 每一步调用 prompter 获取 LLM 动作计划；
4. 解析计划；
5. 调用运动规划和仿真执行；
6. 保存每个 run 的结果。

任务映射：

```python
TASK_NAME_MAP = {
    "sort": SortOneBlockTask,
    "cabinet": CabinetTask,
    "rope": MoveRopeTask,
    "sweep": SweepTask,
    "sandwich": MakeSandwichTask,
    "pack": PackGroceryTask,
}
```

## 2. Prompt 和大模型模块

### `prompting/plan_prompter.py`

集中式规划模式，`comm_mode=plan` 或 `comm_mode=chat` 时使用。

重点函数：

```python
SingleThreadPrompter.query_once()
```

这里需要接入大模型 API。

### `prompting/dialog_prompter.py`

多机器人对话式规划模式，`comm_mode=dialog` 时使用。

重点函数：

```python
DialogPrompter.query_once()
```

初期建议先跑 `plan`，后期有余力再尝试 `dialog`。

## 3. 解析模块

### `prompting/parser.py`

负责把大模型输出文本解析成 `LLMPathPlan`。

支持动作包括：

- `PICK`
- `PLACE`
- `PUT`
- `OPEN`
- `SWEEP`
- `DUMP`
- `MOVE`
- `WAIT`

输出必须包含：

```text
EXECUTE
NAME xxx ACTION xxx
```

对于 `rope` 和 `pack` 等任务，还可能要求 `PATH [(x,y,z), ...]`。

## 4. 反馈模块

### `prompting/feedback.py`

负责在执行前检查计划是否合理：

- 任务约束；
- reachability；
- IK 是否可解；
- 碰撞检测；
- waypoint 是否平滑。

如果失败，会生成反馈并交给大模型重规划。

## 5. 运动规划和执行

### `rocobench/policy.py`

`PlannedPathPolicy` 将高层动作计划转换成具体机械臂动作。

### `rocobench/rrt_multi_arm.py`

多机械臂 RRT 运动规划。

### `rocobench/rrt.py`

基础 RRT、双向 RRT、路径平滑等算法。

## 6. 推荐优先修改位置

优先级最高：

- `prompting/plan_prompter.py::query_once`
- `prompting/dialog_prompter.py::query_once`
- prompt 格式约束和任务策略
- response 清洗逻辑

可以考虑修改：

- parser 容错逻辑；
- feedback 后重试逻辑；
- 特定任务 prompt；
- PATH 生成策略。

谨慎修改：

- RRT 底层算法；
- MuJoCo 环境；
- evaluator。
