
  在 parser 层大概率能解析，但在 sort 任务里会被任务反馈判为不合法，因为 sort 的 action prompt 要求：

  PICK <object name> PLACE <location>

  所以更合法的 sort 输出应该类似：

  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  下面按这个例子讲整个链路。

  ———

  # 1. 模型会收到什么信息？

  模型不是直接看到 MuJoCo，也不是看到图像，而是收到程序拼出来的 文本 prompt。

  默认 comm_mode=plan 时，prompt 在这里拼：

  prompting/plan_prompter.py
  SingleThreadPrompter.compose_system_prompt()

  它主要由几部分组成。

  ———

  ## 1.1 任务背景

  来自：

  env.describe_task_context()

  对于 sort 任务，内容大概是：

  7 panels on the table, ordered left to right: panel1,...,panel7.

  There are 3 cubes, each robot must place their cube on the correct target:

  Alice: blue_square -> panel2
  Bob: pink_polygon -> panel4
  Chad: yellow_trapezoid -> panel6

  Each robot has limited reach:

  Alice can reach panel1, panel2, panel3
  Bob can reach panel3, panel4, panel5
  Chad can reach panel5, panel6, panel7

  也就是说模型知道：

  - 每个机器人负责什么物体；
  - 每个物体目标 panel 是什么；
  - 每个机器人能 reach 哪些 panel。

  ———

  ## 1.2 动作语法说明

  来自：

  env.get_action_prompt()

  sort 任务里是：

  [Action Options]
  1) PICK <object name> PLACE <location>
  2) WAIT

  Only PICK an object if your gripper is empty.
  Target <location> for PLACE should be panel or a bin.

  [Action Output Instruction]
  You must first output 'EXECUTE\n',
  then give exactly one action per robot,
  put each on a new line.

  Example:
  EXECUTE
  NAME Alice ACTION PICK red_square PLACE panel3
  NAME Bob ACTION WAIT
  NAME Chad ACTION PICK green_trapezoid PLACE panel6

  所以模型被明确要求用这个格式。

  ———

  ## 1.3 当前场景状态

  来自：

  obs = env.get_obs()
  obs_desp = env.describe_obs(obs)

  sort 任务里会转成文字，例如：

  [Scene description]
  blue_square is on panel1
  pink_polygon is on panel5
  yellow_trapezoid is on panel7

  Alice's gripper is empty, can reach cubes: blue_square, can't reach cubes: pink_polygon,
  yellow_trapezoid
  Bob's gripper is empty, can reach cubes: pink_polygon, can't reach cubes: blue_square, yellow_trapezoid
  Chad's gripper is empty, can reach cubes: yellow_trapezoid, can't reach cubes: blue_square,
  pink_polygon.

  这里不是固定文本，取决于当前 MuJoCo 状态。

  ———

  ## 1.4 历史和反馈

  如果上一轮失败，程序会把失败原因也塞回 prompt，例如：

  Previous Plans Require Improvement:
  [Environment Feedback]:
  - Previous Plan:
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  - Goal Step Alice (...):
    - IK failed

  或者：

  Collision detected: Alice-Bob

  模型下一轮就会基于这个反馈重新输出动作。

  ———

  ## 1.5 严格输出约束

  在 prompting/llm_client.py 里，程序还会追加：

  Return the final answer in English.
  Do not use markdown code fences.
  Do not output Chinese.
  The executable final plan must contain exactly one EXECUTE block.
  After EXECUTE, each robot must have exactly one line.
  Each line must follow:
  NAME <robot_name> ACTION <valid_action>
  If a robot should do nothing, use WAIT.
  Use only robot names, object names, and actions provided in the prompt.

  所以模型最终收到的核心信息是：

  任务规则
  + 动作格式
  + 当前场景
  + 历史执行反馈
  + 严格输出格式要求

  ———

  # 2. 程序怎么把 prompt 发给模型？

  代码位置：

  prompting/llm_client.py

  核心函数：

  query_ollama_chat()

  它会向 Ollama 发 HTTP 请求：

  POST http://<OLLAMA_BASE_URL>/api/chat

  payload 类似：

  {
      "model": "llama3.3:70b",
      "messages": [
          {
              "role": "system",
              "content": system_prompt + strict_instruction
          },
          {
              "role": "user",
              "content": "Based on the task context, action options, and scene description above, output
  the next valid robot plan now..."
          }
      ],
      "stream": False,
      "think": False,
      "options": {
          "temperature": 0,
          "num_predict": 1024
      }
  }

  所以模型和程序的接口就是：

  Python requests.post()
      ↓
  Ollama /api/chat
      ↓
  llama3.3:70b
      ↓
  返回 message.content

  ———

  # 3. 模型怎么处理这些信息？

  从程序视角看，模型只做一件事：

  根据 prompt 输出下一步高层机器人计划

  它不会直接控制机器人，也不会调用 MuJoCo。

  例如它看到：

  blue_square is on panel1
  Alice can reach blue_square
  blue_square target is panel2
  Alice can reach panel2

  它可能输出：

  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  意思是：

  - Alice 把 blue_square 放到 panel2；
  - Bob 不动；
  - Chad 不动。

  ———

  # 4. 输出里的每个文字是什么意思？

  以合法 sort 输出为例：

  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  逐个解释。

  ———

  ## 4.1 EXECUTE

  这是一个分隔符。

  程序用它判断：

  从这里开始是可执行计划

  parser 会做：

  execute_str = response.split('EXECUTE')[1]

  也就是说，EXECUTE 前面如果有废话，程序会尽量忽略。
  但最好不要输出废话。

  ———

  ## 4.2 NAME

  表示后面跟的是机器人名字。

  例如：

  NAME Alice

  表示这一行是 Alice 的动作。

  机器人名字必须是 prompt 里出现的合法名字：

  Alice
  Bob
  Chad

  它们映射到底层机器人是：

  Alice -> ur5e_robotiq
  Bob   -> panda
  Chad  -> ur5e_suction

  ———

  ## 4.3 ACTION

  表示后面跟的是动作内容。

  parser 会用它切分：

  agent_name = line.split('NAME')[1].split('ACTION')[0].strip()
  action_desp = line.split('ACTION')[1].strip()

  例如：

  NAME Alice ACTION PICK blue_square PLACE panel2

  会解析成：

  agent_name = "Alice"
  action_desp = "PICK blue_square PLACE panel2"

  ———

  ## 4.4 PICK blue_square PLACE panel2

  这是 sort 任务的复合动作。

  含义是：

  Alice 去抓 blue_square，然后放到 panel2

  parser 会把它拆成两个内部计划：

  1. PICK blue_square
  2. PLACE blue_square panel2

  也就是：

  parse_pick_and_place()

  内部会先调用：

  parse_pick_action()

  再构造 place plan。

  ———

  ## 4.5 WAIT

  例如：

  NAME Bob ACTION WAIT

  含义是：

  Bob 保持当前末端位置，不移动，不抓取，不释放

  但注意：WAIT 不是程序完全忽略 Bob。
  程序仍然会为 Bob 生成一个目标位姿：

  Bob 当前夹爪位置 = Bob 的目标位置

  这样 RRT 在多机械臂规划时知道 Bob 应该保持不动。

  ———

  # 5. 模型输出后，文本给谁处理？

  模型返回后，调用链是：

  Ollama response
    ↓
  llm_client.clean_llm_response()
    ↓
  SingleThreadPrompter.prompt_one_round()
    ↓
  LLMResponseParser.parse()
    ↓
  FeedbackManager.give_feedback()
    ↓
  PlannedPathPolicy.plan()
    ↓
  env.step()

  ———

  # 6. 第一层处理：清洗文本

  位置：

  prompting/llm_client.py

  函数：

  clean_llm_response()

  会做：

  1. 去掉 markdown 代码块；
  2. 如果出现多个 EXECUTE，从第一个 EXECUTE 开始截；
  3. 去掉首尾空格。

  例如模型输出：

  Here is the plan:

  ```text
  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT


  会被清洗成：

  ```text
  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT
  ```

  # 7. 第二层处理：parser 解析文本

  位置：

  prompting/parser.py

  核心函数：

  LLMResponseParser.parse(obs, response)

  它先检查：

  必须有 EXECUTE
  必须有 NAME
  必须有 ACTION
  必须每个机器人都有一行
  不能漏 Alice/Bob/Chad
  不能重复机器人

  然后逐行解析。

  ———

  ## 7.1 Alice 行怎么解析？

  输入：

  NAME Alice ACTION PICK blue_square PLACE panel2

  因为同时包含 PICK 和 PLACE，进入：

  parse_pick_and_place()

  里面先处理 PICK blue_square。

  程序检查：

  blue_square 是否在 obs.objects 里
  Alice 当前手里是否为空
  blue_square 是否已经被别的机器人拿着
  blue_square 有没有 grasp site

  对于 sort：

  get_grasp_site("blue_square")

  返回：

  blue_square_top

  然后取 MuJoCo 里的位置：

  pick_pos = obj_state.sites["blue_square_top"].xpos
  pick_quat = obj_state.sites["blue_square_top"].xquat

  形成抓取目标：

  pick_target_pose = [x, y, z, qw, qx, qy, qz]

  然后处理 PLACE panel2。

  程序通过：

  env.get_target_pos("Alice", "panel2")

  找到 panel2 的位置，形成放置目标。

  最终 Alice 的一句话会变成两个内部计划：

  Plan 1: Alice 移动到 blue_square_top 并抓取
  Plan 2: Alice 移动到 panel2 并释放

  ———

  ## 7.2 Bob/Chad 的 WAIT 怎么解析？

  输入：

  NAME Bob ACTION WAIT

  进入：

  parse_wait_action()

  它取 Bob 当前夹爪位姿：

  current_pose = robot_state.ee_pose

  然后构造：

  ee_targets = current_pose
  tograsp = None
  action_strs = "WAIT"

  意思是：

  Bob 的目标就是保持在原地

  Chad 同理。

  ———

  # 8. parser 输出什么？

  parser 最终输出：

  parse_succ, parsed_str, llm_plans

  其中 llm_plans 是一个或多个：

  LLMPathPlan

  对于：

  Alice PICK blue_square PLACE panel2
  Bob WAIT
  Chad WAIT

  因为 Alice 是 pick-and-place，内部可能变成两个 LLMPathPlan：

  LLMPathPlan 1: Alice pick，Bob wait，Chad wait
  LLMPathPlan 2: Alice place，Bob wait，Chad wait

  每个 LLMPathPlan 里有：

  agent_names = ["Alice", "Bob", "Chad"]

  ee_targets = {
      "Alice": 目标末端位姿,
      "Bob": 当前末端位姿,
      "Chad": 当前末端位姿,
  }

  ee_waypoints = {
      "Alice": 中间路径点,
      "Bob": 保持不动路径点,
      "Chad": 保持不动路径点,
  }

  tograsp = {
      "Alice": ("blue_square", "blue_square_top", 1 或 0),
      "Bob": None,
      "Chad": None,
  }

  action_strs = {
      "Alice": "PICK blue_square PLACE panel2",
      "Bob": "WAIT",
      "Chad": "WAIT",
  }

  其中：

  tograsp(..., 1)

  表示抓取；

  tograsp(..., 0)

  表示释放。

  ———

  # 9. 第三层处理：环境反馈检查

  解析成功后，进入：

  FeedbackManager.give_feedback(llm_plan)

  位置：

  prompting/feedback.py

  它会检查这个计划能不能执行。

  ———

  ## 9.1 任务规则检查

  调用：

  env.get_task_feedback(llm_plan, pose_dict)

  sort 任务里会检查：

  ### 第一，不能只有 PICK 没有 PLACE

  所以你原来的例子：

  NAME Alice ACTION PICK blue_square

  会被判失败：

  Alice's ACTION must contain both PICK and PLACE

  ### 第二，不能放错目标

  比如：

  NAME Alice ACTION PICK blue_square PLACE panel6

  会失败，因为：

  blue_square 目标是 panel2

  代码里有：

  correct_panel = self.cube_to_bin[obj]
  if correct_panel not in target:
      feedback += ...

  ### 第三，不能全部 WAIT

  NAME Alice ACTION WAIT
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  会失败，因为任务还没完成。

  ———

  ## 9.2 可达性检查

  检查目标点是否在机器人 reach range 内。

  例如 Alice 只能 reach：

  panel1, panel2, panel3

  如果让 Alice 去 panel6，可能被判：

  Reachability failed

  ———

  ## 9.3 IK 检查

  即：

  这个末端位姿能不能转成机械臂关节角

  代码：

  self.planner.inverse_kinematics_all(...)

  如果 IK 解不出来，会反馈给模型：

  IK failed

  ———

  ## 9.4 碰撞检查

  检查目标姿态是否和其他机器人、物体、桌子碰撞。

  如果碰撞，会生成：

  Collision detected: Alice-Bob

  ———

  # 10. 如果检查失败怎么办？

  如果反馈失败，程序不会直接执行。

  它会把失败原因加到下一轮 prompt 里，再问模型：

  Previous Plans Require Improvement:
  [Environment Feedback]:
  ...

  这个循环在：

  SingleThreadPrompter.prompt_one_round()

  默认最多重试：

  --num_replans 5

  所以流程是：

  模型输出
    ↓
  parser/feedback 失败
    ↓
  失败原因加入 prompt
    ↓
  再次问模型

  直到：

  ready_to_execute = True

  或者重试次数用完。

  ———

  # 11. 检查通过后，机器怎么执行？

  一旦计划通过，进入：

  PlannedPathPolicy

  位置：

  rocobench/policy.py

  代码：

  policy = PlannedPathPolicy(..., path_plan=plan)
  plan_success, reason = policy.plan(env)

  ———

  ## 11.1 高层动作变成末端目标

  比如：

  Alice PICK blue_square PLACE panel2

  已经被 parser 转成了：

  Alice 的抓取末端位姿
  Alice 的放置末端位姿
  Bob 当前位姿
  Chad 当前位姿

  ———

  ## 11.2 末端目标变成关节角

  程序调用 IK：

  parse_llm_plan_to_qpos()

  把：

  末端执行器位置 + 姿态

  转成：

  机器人各关节角 qpos

  ———

  ## 11.3 RRT 规划无碰撞路径

  调用：

  MultiArmRRT.plan()

  输入：

  当前关节角
  目标关节角
  中间 waypoint
  碰撞约束
  抓取物体信息

  输出：

  一串关节路径

  例如概念上：

  qpos_0 -> qpos_1 -> qpos_2 -> ... -> qpos_goal

  ———

  ## 11.4 关节路径变成控制命令

  每个 qpos 会转成：

  SimAction

  里面有：

  ctrl_idxs
  ctrl_vals
  qpos_idxs
  qpos_target
  eq_active_idxs
  eq_active_vals

  简单理解：

  ctrl_idxs: 控制哪些关节/夹爪
  ctrl_vals: 控制目标值
  eq_active: 是否激活物体和夹爪的 weld 约束

  对于 PICK：

  tograsp = ("blue_square", "blue_square_top", 1)

  会生成抓取动作：

  夹爪闭合 / suction 激活 / weld 激活

  对于 PLACE：

  tograsp = ("blue_square", "blue_square_top", 0)

  会生成释放动作：

  夹爪张开 / suction 关闭 / weld 解除

  ———

  # 12. MuJoCo 怎么执行？

  执行在：

  env.step(sim_action)

  位置：

  rocobench/envs/base_env.py

  核心流程：

  for step in range(self.sim_forward_steps):
      self.data.ctrl[ctrl_idxs] = ctrl_vals

      if eq_active_idxs is not None:
          self.physics.model.eq_active[eq_active_idxs] = eq_active_vals

      self.physics.step()

  也就是说：

  1. 给机器人关节发送目标控制值；
  2. 如果是抓取/释放，修改 weld 约束；
  3. MuJoCo 物理仿真往前推进；
  4. 渲染视频帧；
  5. 更新 observation。

  执行完成后重新获取状态：

  next_obs = self.get_obs()

  然后判断：

  reward, done = self.get_reward_done(next_obs)

  ———

  # 13. 怎么判断任务是否完成？

  每个任务自己定义完成条件。

  sort 任务在：

  rocobench/envs/task_sort.py
  get_reward_done()

  逻辑是：

  reward = 1
  done = True

  for block_name in ["blue_square", "pink_polygon", "yellow_trapezoid"]:
      correct_bin = self.cube_to_bin[block_name]

      if block 不在正确 panel 附近:
          reward = 0
          done = False
          break

  return reward, done

  具体对应：

  blue_square 必须在 panel2
  pink_polygon 必须在 panel4
  yellow_trapezoid 必须在 panel6

  如果三个都放对了：

  reward = 1
  done = True

  否则：

  reward = 0
  done = False

  ———

  # 14. 一个完整闭环例子

  假设当前场景是：

  blue_square is on panel1
  pink_polygon is on panel5
  yellow_trapezoid is on panel7
  Alice can reach blue_square
  Bob can reach pink_polygon
  Chad can reach yellow_trapezoid

  模型输出：

  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION PICK pink_polygon PLACE panel4
  NAME Chad ACTION PICK yellow_trapezoid PLACE panel6

  程序执行：

  1. parser 检查格式
  2. parser 把每行动作转成 LLMPathPlan
  3. feedback 检查：
     - 每个目标 panel 是否正确
     - 每个机器人是否够得到
     - IK 是否成功
     - 是否碰撞
  4. RRT 规划三台机械臂路径
  5. MuJoCo 执行：
     - Alice 抓 blue_square 放 panel2
     - Bob 抓 pink_polygon 放 panel4
     - Chad 抓 yellow_trapezoid 放 panel6
  6. env.get_reward_done() 检查三个物体是否都到正确 panel
  7. 如果都对，done=True, reward=1
  8. run_dialog.py 写 success=True

  ———

  # 15. 如果用你原来的输出会发生什么？

  你的输出是：

  EXECUTE
  NAME Alice ACTION PICK blue_square
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  它的流程大概是：

  ## parser 阶段

  可能可以解析：

  Alice 去抓 blue_square
  Bob wait
  Chad wait

  ## feedback 阶段

  sort 任务会检查到：

  Alice's ACTION must contain both PICK and PLACE

  因为 sort 任务要求一步输出：

  PICK object PLACE target

  所以它会失败，不执行。

  然后程序把反馈发回模型，让模型重新输出，例如：

  EXECUTE
  NAME Alice ACTION PICK blue_square PLACE panel2
  NAME Bob ACTION WAIT
  NAME Chad ACTION WAIT

  ———

  # 16. 总结一句话

  这个系统里：

  模型只负责输出高层文本计划

  例如：

  NAME Alice ACTION PICK blue_square PLACE panel2

  程序负责：

  解析文本
  → 检查任务规则
  → 检查可达性/IK/碰撞
  → 转成末端位姿
  → 转成关节路径
  → 生成 MuJoCo 控制命令
  → 执行仿真
  → 判断是否完成任务

  所以模型输出的每个词不是随便写的，而是一个简易 DSL：

  EXECUTE                 开始可执行计划
  NAME Alice              指定机器人 Alice
  ACTION                  后面是动作
  PICK blue_square        抓 blue_square
  PLACE panel2            放到 panel2
  WAIT                    保持不动

  然后这段 DSL 被 parser.py 变成 LLMPathPlan，再被 policy.py 和 rrt_multi_arm.py 变成机器人实际运动。