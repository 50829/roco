# 多机器人协同的具身操作：项目描述

## 1. 项目背景

本项目基于 RocoBench / RoCo 多机器人协作仿真框架，要求我们在 MuJoCo 多机器人操作环境中接入大模型，让大模型根据任务目标和环境观测生成多机器人协作动作计划，并通过已有的解析器、反馈模块、运动规划器和仿真环境执行这些动作。

本项目的核心不是从零训练机械臂，也不是做强化学习，而是：

> 让大模型稳定地输出可解析、可执行、符合物理约束的多机器人协作计划，从而提高任务成功率。

## 2. 系统整体流程

```text
MuJoCo / RocoBench 环境
    ↓
获取当前观测：物体位置、机器人状态、任务规则
    ↓
构造 prompt
    ↓
调用本地或远程大模型
    ↓
大模型输出动作计划
    ↓
Parser 解析为结构化 LLMPathPlan
    ↓
FeedbackManager 检查任务约束、IK、碰撞和路径
    ↓
RRT / MultiArmRRT 规划机械臂运动路径
    ↓
PlannedPathPolicy 执行动作
    ↓
MuJoCo 仿真更新状态
    ↓
判断任务是否成功
```

## 3. 需要完成的任务

评测包含 6 类多机器人具身操作任务：

| 任务名 | 代码参数 | 任务描述 |
|---|---|---|
| Sort Cubes | `sort` | 多机器人将方块分类到对应区域 |
| Arrange Cabinet | `cabinet` | 多机器人打开柜门、取出杯子并放到正确杯垫 |
| Move Rope | `rope` | 两机器人协同移动绳子越过障碍并放入目标槽 |
| Sweep Floor | `sweep` | 两机器人使用扫帚、簸箕和垃圾桶完成清扫 |
| Make Sandwich | `sandwich` | 两机器人按正确顺序堆叠三明治食材 |
| Pack Grocery | `pack` | 两机器人将杂货协同装入箱子 |

## 4. 项目目标

### 最低目标

1. 环境可以正常运行；
2. `run_dialog.py` 可以调用大模型；
3. 至少能在简单任务上完成若干成功 run；
4. 可以运行 `evaluator.py` 得到完整评测结果。

### 主要目标

1. 提高 6 个任务的总成功率；
2. 降低格式错误、解析错误、IK 失败、碰撞和超时；
3. 总结有效 prompt、协作机制和路径规划改进；
4. 形成可复现的代码和报告材料。

### 最终验收方式

助教会进入实例运行：

```bash
python evaluator.py
```

因此最终代码和环境需要保证可以复现运行。

## 5. 当前仓库注意事项

当前代码中，核心大模型调用处仍是 TODO，需要我们实现：

- `prompting/plan_prompter.py::query_once`
- `prompting/dialog_prompter.py::query_once`

另外，当前本地仓库检查发现 `rocobench/envs` 目录缺失。如果运行时报：

```text
ModuleNotFoundError: No module named 'rocobench.envs'
```

说明代码包不完整，需要从官方镜像或完整压缩包中补齐环境、任务文件和 MuJoCo assets。
