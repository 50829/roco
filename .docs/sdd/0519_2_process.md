下面按 从 python evaluator.py 开始，把这个仓库的执行链路、大模型接口、数据流完整讲一遍。

  ———

  # 1. 总入口：evaluator.py

  你实际跑 evaluate 时，一般是：

  conda activate roco
  cd /inspire/hdd/project/26summer-camp-09/26220478/code
  python evaluator.py

  或者无显示环境：

  xvfb-run -a python evaluator.py

  evaluator.py 的作用不是直接做仿真，而是 批量调度 run_dialog.py。

  核心逻辑在：

  results.append(test_run_dialog("sort", 5, output_root))
  results.append(test_run_dialog("cabinet", 5, output_root))
  results.append(test_run_dialog("rope", 5, output_root))
  results.append(test_run_dialog("sweep", 5, output_root))
  results.append(test_run_dialog("sandwich", 5, output_root))
  results.append(test_run_dialog("pack", 5, output_root))

  也就是说它依次跑 6 个任务，每个任务 5 次。

  ———

  # 2. evaluator.py 怎么调用单任务

  每个任务会进入：

  test_run_dialog(task, num_runs, output_dir)

  里面拼出一个子进程命令：

  command = [
      sys.executable,
      'run_dialog.py',
      '--task', task,
      '--run_name', run_name,
      '--data_dir', task_output_dir,
      '--start_id', str(-1),
      '--num_runs', str(num_runs),
      '--skip_display',
      '--tsteps', str(10),
      '--seed', str(seed),
      '--run_timeout', str(run_timeout),
  ]

  所以实际执行类似：

  python run_dialog.py \
    --task sort \
    --run_name runs \
    --data_dir <某个输出目录> \
    --start_id -1 \
    --num_runs 5 \
    --skip_display \
    --tsteps 10 \
    --seed 0 \
    --run_timeout 600

  注意：evaluator.py 没有显式传 --llm_source，所以模型名靠环境变量：

  export ROCO_LLM_MODEL=llama3.3:70b

  否则默认也是：

  llama3.3:70b

  ———

  # 3. run_dialog.py 的执行流程

  run_dialog.py 是真正的单任务主程序。

  ## 3.1 参数解析

  它会读这些参数：

  --task
  --num_runs
  --run_name
  --data_dir
  --tsteps
  --comm_mode
  --llm_source
  --run_timeout
  ...

  默认：

  --comm_mode plan
  --llm_source os.environ.get("ROCO_LLM_MODEL", "llama3.3:70b")

  所以如果你设了：

  export ROCO_LLM_MODEL=llama3.3:70b

  程序就会用这个 Ollama 模型名。

  ———

  ## 3.2 创建任务环境

  在 main(args) 里：

  TASK_NAME_MAP = {
      "sort": SortOneBlockTask,
      "cabinet": CabinetTask,
      "rope": MoveRopeTask,
      "sweep": SweepTask,
      "sandwich": MakeSandwichTask,
      "pack": PackGroceryTask,
  }

  例如：

  python run_dialog.py --task sort

  会创建：

  env = SortOneBlockTask(...)

  这个 env 是 MuJoCo 仿真环境，负责：

  1. 维护机器人和物体状态；
  2. 生成 observation；
  3. 给大模型描述任务和场景；
  4. 执行动作；
  5. 判断任务是否成功。

  ———

  ## 3.3 特殊任务配置

  rope 和 pack 会自动改配置：

  if args.task == 'rope':
      args.output_mode = 'action_and_path'
      args.split_parsed_plans = True
      args.control_freq = 20
      args.max_failed_waypoints = 0

  elif args.task == 'pack':
      args.output_mode = 'action_and_path'
      args.control_freq = 10
      args.split_parsed_plans = True
      args.max_failed_waypoints = 0

  也就是说：

  - 普通任务只需要大模型输出动作；
  - rope / pack 还需要大模型输出路径 PATH [(x,y,z), ...]。

  ———

  # 4. LLMRunner 是核心调度器

  run_dialog.py 会创建：

  runner = LLMRunner(...)
  runner.run(args)

  LLMRunner 里面初始化了几类核心对象：

  ## 4.1 Parser

  self.parser = LLMResponseParser(...)

  负责把大模型文本解析成结构化动作。

  例如模型输出：

  EXECUTE
  NAME Alice ACTION PICK blue_square
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  会被解析成 LLMPathPlan 对象。

  ———

  ## 4.2 FeedbackManager

  self.feedback_manager = FeedbackManager(...)

  负责检查大模型计划是否可执行，包括：

  1. 任务约束；
  2. reachability；
  3. IK 是否能解；
  4. 是否碰撞；
  5. waypoint 是否合理。

  如果不合理，会生成反馈，再让大模型重规划。

  ———

  ## 4.3 MultiArmRRT

  self.planner = MultiArmRRT(...)

  底层多机械臂运动规划器。

  大模型只给高层动作，比如：

  Alice PICK blue_square

  真正机械臂怎么动，是靠 MultiArmRRT 算路径。

  ———

  ## 4.4 Prompter

  默认 comm_mode=plan，所以会用：

  SingleThreadPrompter

  代码位置：

  prompting/plan_prompter.py

  如果设置：

  --comm_mode dialog

  会用：

  DialogPrompter

  代码位置：

  prompting/dialog_prompter.py

  目前建议先用默认 plan。

  ———

  # 5. 每个 episode 怎么跑

  核心在：

  LLMRunner.one_run()

  一个 run_0 / run_1 就是一次 episode。

  大致流程：

  one_run()
    ├── env.seed()
    ├── env.reset()
    ├── obs = env.get_obs()
    ├── for step in range(tsteps):
    │     ├── 保存当前环境 env_init.pkl
    │     ├── 调用大模型生成计划
    │     ├── 解析大模型输出
    │     ├── 做环境反馈检查
    │     ├── RRT 运动规划
    │     ├── MuJoCo 执行动作
    │     ├── 保存 env_end.pkl / video / json
    │     └── 判断 done
    └── 写 stepsX_success_True/False.json

  ———

  # 6. 大模型交互接口是什么？

  这个仓库现在已经接成了 Ollama HTTP API。

  接口代码在：

  prompting/llm_client.py

  核心函数：

  query_ollama_chat(...)

  它请求的是：

  url = f"{_ollama_base_url()}/api/chat"

  也就是：

  http://<ollama-host>:11434/api/chat

  如果你设置：

  export OLLAMA_BASE_URL=http://GPU机器IP:11434

  程序就会请求：

  http://GPU机器IP:11434/api/chat

  如果不设置，默认：

  http://127.0.0.1:11434/api/chat

  ———

  ## 6.1 发送给 Ollama 的 payload

  代码里构造的是：

  payload = {
      "model": model,
      "messages": [
          {
              "role": "system",
              "content": system_prompt + STRICT_EXECUTE_INSTRUCTION
          },
          {
              "role": "user",
              "content": user_prompt or DEFAULT_USER_PROMPT
          },
      ],
      "stream": False,
      "think": False,
      "options": {
          "temperature": temperature,
          "num_predict": max_tokens,
      },
  }

  所以大模型交互本质就是：

  Python 程序
    -> HTTP POST /api/chat
    -> Ollama
    -> llama3.3:70b
    -> 返回 message.content

  Python 代码里没有直接加载 70B 模型，也不直接管 GPU。GPU 是 Ollama 那边负责。

  ———

  ## 6.2 模型返回后怎么处理

  Ollama 返回 JSON，代码取：

  content = data.get("message", {}).get("content", "")

  然后做清洗：

  clean_llm_response(content)

  清洗逻辑包括：

  1. 去掉 markdown 代码块；
  2. 如果有多个 EXECUTE，从第一个 EXECUTE 开始截取；
  3. 去掉首尾空白。

  ———

  # 7. Prompt 是怎么组成的？

  默认 plan 模式下，走：

  prompting/plan_prompter.py

  关键函数：

  SingleThreadPrompter.compose_system_prompt()

  它把这些东西拼成 system prompt：

  ## 7.1 任务说明

  来自环境：

  task_desp = self.env.describe_task_context()

  比如 sort 任务会告诉模型：

  - 有哪些机器人；
  - 目标是什么；
  - 哪些物体要放到哪个 panel。

  ———

  ## 7.2 动作说明

  来自环境：

  action_desp = self.env.get_action_prompt()

  这里会告诉模型可以输出什么动作，例如：

  PICK
  PLACE
  PUT
  OPEN
  SWEEP
  DUMP
  MOVE
  WAIT

  ———

  ## 7.3 当前观测

  来自：

  obs = env.get_obs()
  obs_desp = self.env.describe_obs(obs)

  这里会把 MuJoCo 里的状态转成文本，例如：

  - 机器人当前夹爪位置；
  - 每个机器人手里有没有东西；
  - 物体当前位置；
  - panel 位置；
  - door 状态等。

  ———

  ## 7.4 历史与失败反馈

  如果之前执行失败，会拼进去：

  self.failed_plans
  plan_feedbacks

  例如：

  Previous Plans Require Improvement:
  IK failed...
  Collision detected...
  Out of reach...

  ———

  ## 7.5 最终指令

  最后拼上：

  get_plan_prompt(self.env)

  里面要求模型：

  Reason about the task step-by-step...
  Propose a plan of exactly one action per robot.
  Strictly follow [Action Output Instruction].

  然后 llm_client.py 还会额外附加强约束：

  Return the final answer in English.
  Do not use markdown code fences.
  Do not output Chinese.
  The executable final plan must contain exactly one EXECUTE block.
  After EXECUTE, each robot must have exactly one line.
  Each line must follow: NAME <robot_name> ACTION <valid_action>
  If a robot should do nothing, use WAIT.

  ———

  # 8. 大模型输出格式

  普通任务期望：

  EXECUTE
  NAME Alice ACTION PICK blue_square
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  rope / pack 这类 action_and_path 任务可能需要：

  EXECUTE
  NAME Alice ACTION MOVE PATH [(0.1,0.2,0.5), (0.2,0.2,0.5)]
  NAME Bob ACTION WAIT PATH [(0.3,0.1,0.5), (0.3,0.1,0.5)]
  NAME Chad ACTION WAIT PATH [(0.4,0.1,0.5), (0.4,0.1,0.5)]

  ———

  # 9. 大模型返回后，数据怎么流动？

  模型返回文本后进入：

  self.parser.parse(obs, response)

  代码位置：

  prompting/parser.py

  它做几件事：

  ## 9.1 检查关键词

  必须包含：

  EXECUTE
  NAME
  ACTION

  如果是 path 模式，还要包含：

  PATH

  ———

  ## 9.2 检查每个机器人都有动作

  比如环境里有：

  Alice
  Bob
  Chad

  那模型必须输出三行。

  少一个机器人，解析失败。

  ———

  ## 9.3 解析单行动作

  比如：

  NAME Alice ACTION PICK blue_square

  会进入：

  parse_pick_action()

  如果是：

  NAME Bob ACTION PLACE panel2

  会进入：

  parse_place_action()

  如果是：

  NAME Chad ACTION WAIT

  会进入：

  parse_wait_action()

  ———

  ## 9.4 生成 LLMPathPlan

  解析成功后，会生成：

  LLMPathPlan

  它里面包含：

  agent_names
  ee_targets
  ee_waypoints
  tograsp
  inhand
  action_strs
  parsed_proposal
  return_home

  也就是说，文本计划被转成了结构化计划。

  ———

  # 10. 环境反馈检查

  解析成功后，会进入：

  self.feedback_manager.give_feedback(llm_plan)

  位置：

  prompting/feedback.py

  它会检查：

  ## 10.1 task feedback

  self.env.get_task_feedback(...)

  检查任务规则，比如：

  - 这个物体能不能被这个机器人抓；
  - 目标是不是合法；
  - 是否违反任务约束。

  ———

  ## 10.2 reachability

  self.env.check_reach_range(...)

  检查目标点是否在机器人可达范围内。

  ———

  ## 10.3 IK

  self.planner.inverse_kinematics_all(...)

  检查目标末端位姿能不能反解成关节角。

  ———

  ## 10.4 collision

  self.planner.get_collided_links(...)

  检查目标姿态是否碰撞。

  ———

  ## 10.5 path feedback

  对于 rope / pack，还会检查路径点：

  - waypoint 是否 IK 可解；
  - waypoint 是否碰撞；
  - waypoint 间距是否过于不均匀。

  ———

  # 11. 如果反馈失败怎么办？

  在 SingleThreadPrompter.prompt_one_round() 里有重规划循环：

  for i in range(self.num_replans):
      response = self.query_once(...)
      parse_succ = self.parser.parse(...)
      ready_to_execute = self.feedback_manager.give_feedback(...)
      if ready_to_execute:
          break

  默认 run_dialog.py 里：

  --num_replans 5

  所以一次 high-level step 最多会让模型改 5 次。

  数据流是：

  LLM 输出计划
    -> parser 解析失败 / feedback 检查失败
    -> 生成文字反馈
    -> 加到下一轮 prompt
    -> 再问 LLM

  ———

  # 12. 计划通过后，怎么执行？

  如果：

  ready_to_execute == True

  则进入运动规划：

  policy = PlannedPathPolicy(...)
  plan_success, reason = policy.plan(env)

  代码位置：

  rocobench/policy.py

  这里做的事情是：

  ## 12.1 把末端目标转成关节目标

  parse_llm_plan_to_qpos()

  把大模型给的目标末端位姿：

  end-effector pose

  转成：

  robot joint qpos

  ———

  ## 12.2 RRT 找路径

  self.rrt_planner.plan(...)

  输入：

  当前关节角
  目标关节角
  中间 waypoint
  碰撞约束
  手里抓着的物体

  输出：

  一串关节路径 qpos

  ———

  ## 12.3 生成低层控制动作

  map_qpos_to_ctrl()

  把 qpos 路径转成 MuJoCo 控制指令：

  SimAction(
      ctrl_idxs=...,
      ctrl_vals=...,
      qpos_idxs=...,
      qpos_target=...,
  )

  如果是 PICK / PLACE，还会生成夹爪或者 weld 约束动作：

  get_grasp_action()

  ———

  # 13. MuJoCo 怎么执行动作？

  回到 run_dialog.py：

  while not policy.plan_exhausted:
      sim_action = policy.act(obs, env.physics)
      obs, reward, done, info = env.step(sim_action)

  env.step() 在：

  rocobench/envs/base_env.py

  它做：

  1. 设置控制量 ctrl；
  2. 设置抓取约束 eq_active；
  3. MuJoCo 往前 step；
  4. 渲染相机；
  5. 保存中间状态；
  6. 重新生成 observation；
  7. 调用 get_reward_done() 判断是否完成。

  ———

  # 14. 每一步会保存什么数据？

  每个 run 的目录类似：

  run_0/
    step_0/
      env_init.pkl
      prompts/
        replan0_xxx.json
        replan0_feedback_xxx.json
      llm_plan_0.pkl
      rrt_plan_0.pkl
      actions_0.pkl
      execute.mp4
      env_end.pkl
    step_1/
      ...
    steps2_success_False.json
    steps2_success_False.html

  其中：

  ## prompts/*.json

  保存：

  - system prompt；
  - user prompt；
  - LLM response；
  - usage；
  - feedback。

  这是分析模型输出最重要的位置。

  ———

  ## llm_plan_*.pkl

  保存解析后的 LLMPathPlan。

  ———

  ## rrt_plan_*.pkl

  保存 RRT 规划出的关节路径。

  ———

  ## actions_*.pkl

  保存实际低层控制动作。

  ———

  ## execute.mp4

  执行视频。

  ———

  ## stepsX_success_True/False.json

  最终结果：

  {
    "step": 2,
    "success": false,
    "timed_out": false,
    "elapsed_time": 123.4
  }

  evaluator.py 最后就是扫这些 json 来统计成功率。

  ———

  # 15. 总体数据流图

  可以理解成：

  evaluator.py
    |
    | subprocess
    v
  run_dialog.py
    |
    | 创建 env
    v
  MuJoCo Env
    |
    | env.get_obs()
    v
  EnvState 结构化状态
    |
    | env.describe_obs()
    v
  文本 observation
    |
    | compose_system_prompt()
    v
  Prompt
    |
    | HTTP POST /api/chat
    v
  Ollama llama3.3:70b
    |
    | message.content
    v
  LLM 文本计划
    |
    | LLMResponseParser.parse()
    v
  LLMPathPlan
    |
    | FeedbackManager.give_feedback()
    | 若失败：feedback 加回 prompt 重新问模型
    v
  PlannedPathPolicy
    |
    | IK + MultiArmRRT
    v
  低层 SimAction 序列
    |
    | env.step()
    v
  MuJoCo 状态更新
    |
    | get_reward_done()
    v
  success / failure
    |
    v
  写日志、json、html、mp4

  ———

  # 16. 程序和大模型之间的边界

  非常关键：

  ## Python 程序负责

  1. 构造任务环境；
  2. 把环境状态转成 prompt；
  3. 调 Ollama API；
  4. 解析模型输出；
  5. 做可达性 / IK / 碰撞检查；
  6. 做 RRT 运动规划；
  7. 执行 MuJoCo 仿真；
  8. 统计成功率。

  ## 大模型只负责

  根据文本 prompt 输出下一步高层动作：

  EXECUTE
  NAME Alice ACTION ...
  NAME Bob ACTION ...
  NAME Chad ACTION ...

  大模型不直接控制机器人，不直接访问 MuJoCo，也不直接看图片。它看到的是程序生成的文本 observation。

  ———

  # 17. GPU 机器和共享磁盘在这里的关系

  如果 Ollama 在 GPU 机器：

  当前 evaluate 程序
    --HTTP-->
  GPU 机器 Ollama
    --跑 llama3.3:70b-->
  返回文本

  共享磁盘只负责：

  1. 放代码；
  2. 放输出日志；
  3. 放 run artifacts；
  4. 可能放 Ollama 模型文件。

  但是 程序不是通过共享磁盘和模型交互的，而是通过：

  OLLAMA_BASE_URL=http://GPU机器IP:11434

  ———

  # 18. 最小单任务执行链路示例

  你跑：

  python run_dialog.py \
    --task sort \
    --num_runs 1 \
    --run_name debug_sort \
    --data_dir /inspire/hdd/project/26summer-camp-09/rocobench_runs/data \
    --skip_display \
    --comm_mode plan \
    --llm_source llama3.3:70b

  实际发生：

  1. 创建 SortOneBlockTask
  2. reset 环境
  3. 读取当前 block / panel / robot 状态
  4. 生成 prompt
  5. POST http://<OLLAMA_BASE_URL>/api/chat
  6. llama3.3:70b 返回 EXECUTE block
  7. parser 检查格式
  8. feedback 检查 reach / IK / collision
  9. RRT 规划机械臂轨迹
  10. MuJoCo 执行
  11. 判断是否成功
  12. 写 run_0/step_x 日志

  ———

  # 19. 调试时最该看的文件

  如果模型格式错，看：

  run_x/step_y/prompts/replan0_*.json

  如果环境反馈失败，看：

  run_x/step_y/prompts/replan0_feedback_*.json

  如果 RRT 失败，看子进程 stdout：

  output/run_YYYYMMDD_HHMMSS/tasks/01_sort/stdout.log

  如果任务最终失败，看：

  stepsX_success_False.json
  stepsX_success_False.html

  如果完整 evaluate 统计，看：

  output/run_YYYYMMDD_HHMMSS/summary.json

  ———

  一句话总结：
  这个仓库的核心闭环是 MuJoCo 状态 → 文本 prompt → Ollama 70B 输出高层动作 → parser/feedback 检查 → RRT 转
  成低层动作 → MuJoCo 执行 → 再生成新状态。大模型接口就是 prompting/llm_client.py 里对 Ollama /api/chat 的
  HTTP POST。