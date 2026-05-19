# 失败分析与解决方案

## 1. 大模型输出格式错误

### 表现

- 缺少 `EXECUTE`；
- 少了某个机器人；
- 输出 Markdown；
- 输出中文解释；
- 动作不是合法动作。

### 解决

- 强化 prompt 格式约束；
- 降低 temperature；
- 对 response 做清洗；
- 失败时让模型重试。

## 2. Parser 解析失败

### 表现

- `Response does not contain NAME/ACTION`；
- `Object xxx does not exist`；
- `Action cannot be parsed`；
- `PATH does not fit desired format`。

### 解决

- 要求模型只使用观测中的物体名；
- 在 prompt 中列出合法动作；
- 对 PATH 使用固定格式；
- 尽量不要让模型创造新目标名。

## 3. IK 失败

### 表现

- Feedback 中出现 `IK failed`；
- 目标位姿不可达。

### 解决

- 换更近的机器人执行；
- 使用中间 `MOVE`；
- PATH 点 z 方向抬高；
- 减少复杂同时动作。

## 4. 碰撞失败

### 表现

- Feedback 中出现 `Collision detected`；
- RRT 找不到路径；
- 执行中环境 rewind。

### 解决

- 让非必要机器人 `WAIT`；
- 避免双臂同时进入中心区域；
- 增加高位中间点；
- 分步骤执行，而不是一次做太多。

## 5. 超时失败

### 表现

- 单 run 超过 600 秒；
- evaluator 统计 timeout。

### 解决

- 减少模型调用次数；
- 优先使用 `comm_mode=plan`；
- 降低 `num_replans`；
- 避免复杂 PATH；
- 对简单任务使用更直接策略。
