# 我们要做什么

## 1. 一句话目标

> 在 48 小时内接入 80B 以下大模型，跑通 RocoBench 多机器人任务，并通过 prompt、协作策略、反馈重规划和必要代码改进提高 `evaluator.py` 的成功率。

## 2. 具体工作清单

### P0：先让系统跑起来

- [ ] 确认完整代码存在，尤其是 `rocobench/envs/`；
- [ ] 确认 conda 环境、MuJoCo、依赖安装正确；
- [ ] 确认 `python run_dialog.py` 可以运行到大模型调用处；
- [ ] 确认可以用 `xvfb-run` 在无显示环境中运行。

验证命令：

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --debug_mode
```

### P1：部署并接入大模型

- [ ] 选择 80B 以下模型，例如 Llama 3.3 70B、Qwen 72B Instruct 等；
- [ ] 用 Ollama / vLLM / SGLang 等方式部署；
- [ ] 暴露 OpenAI-compatible API；
- [ ] 在 `plan_prompter.py` 中实现 `query_once()`；
- [ ] 必要时在 `dialog_prompter.py` 中实现 `query_once()`。

推荐先实现集中式规划：

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --comm_mode plan
```

### P2：保证输出格式稳定

LLM 最终必须输出类似：

```text
EXECUTE
NAME Alice ACTION PICK red_cube
NAME Bob ACTION WAIT
NAME Chad ACTION PICK blue_cube
```

需要解决：

- 缺少 `EXECUTE`；
- 少机器人；
- 机器人名写错；
- 物体名写错；
- 输出中文解释；
- Markdown 代码块影响解析；
- `PATH` 格式不合法。

建议在模型返回后做轻量清洗：

- 去掉 Markdown 代码块；
- 截取 `EXECUTE` 之后内容；
- 保证返回非空字符串；
- 对明显缺失的 WAIT 做保守补齐，谨慎使用。

### P3：逐任务优化成功率

建议优先级：

```text
sort → sweep → cabinet → sandwich → pack → rope
```

原因：

- `sort` 最简单，适合验证系统；
- `sweep` 动作空间清晰；
- `cabinet` 需要多机器人分工；
- `sandwich` 需要正确顺序；
- `pack` 和 `rope` 需要路径点，难度最高。

### P4：记录实验并形成报告

每次实验都要记录：

- 使用模型；
- 运行命令；
- 任务名称；
- 成功率；
- 失败原因；
- 修改内容；
- 下一步计划。

最终报告需要说明：

- 模型部署方式；
- 大模型接口实现；
- prompt 设计；
- 多机器人协作机制；
- 改进方法；
- 实验结果；
- 失败案例分析；
- 成员分工。

## 3. 不建议做的事情

- 不要依赖修改 `evaluator.py` 提分，最终评测时它可能会被替换；
- 不要一开始就做 finetune，48 小时内性价比低；
- 不要直接跑全量评测调参，先单任务小规模跑通；
- 不要让模型自由长篇解释，parser 更需要严格格式。
