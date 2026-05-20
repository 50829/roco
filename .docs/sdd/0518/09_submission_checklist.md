# 提交检查表

## 1. 代码检查

- [ ] `prompting/plan_prompter.py::query_once` 已实现；
- [ ] 如果使用 `dialog`，`prompting/dialog_prompter.py::query_once` 已实现；
- [ ] 不依赖个人绝对路径；
- [ ] 不提交大模型权重；
- [ ] 不提交 `output/`、`data/`、大日志；
- [ ] `pack_code.sh` 可以打包。

## 2. 环境检查

- [ ] conda 环境可激活；
- [ ] MuJoCo 可用；
- [ ] `rocobench.envs` 可 import；
- [ ] 模型服务已启动；
- [ ] API 地址正确；
- [ ] Jupyter server 保持运行。

## 3. 运行检查

最终至少跑一次：

```bash
python evaluator.py
```

保存：

- `summary.json`；
- 终端截图；
- 关键任务成功截图或 html；
- 失败分析记录。

## 4. 报告检查

小组报告包含：

- [ ] 项目目标；
- [ ] 系统流程；
- [ ] 模型部署；
- [ ] 接口实现；
- [ ] prompt 设计；
- [ ] 改进方法；
- [ ] 实验结果；
- [ ] 失败分析；
- [ ] 截图材料。

个人报告包含：

- [ ] 个人负责模块；
- [ ] 具体代码或实验贡献；
- [ ] 遇到的问题；
- [ ] 解决方法。
