import os
import json
import pickle
import requests
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
import openai
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .llm_client import query_ollama_chat
from typing import List, Tuple, Dict, Union, Optional, Any

PATH_PLAN_INSTRUCTION="""
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach.
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects.
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)]
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""



def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan.
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt


def get_plan_prompt(env: MujocoSimEnv):
    if hasattr(env, "central_plan_prompt"):
        try:
            task_prompt = env.central_plan_prompt()
        except TypeError:
            task_prompt = env.central_plan_prompt([])
        if task_prompt:
            return task_prompt

    return """
Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Output only the final EXECUTE block, with exactly one ACTION line for each robot.
Your final plan output is:
    """


class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan,
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,
        temperature: float = 0,
        max_tokens: int = 1000,
        llm_source: str = "gpt-4",
    ):
        self.env = env
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]


    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[Recent Executed History]\n"
        # Full raw LLM history pollutes the prompt and can make the model copy
        # invalid old plans.  Keep only the latest few structured execution
        # summaries.
        for i, history in enumerate(self.round_history[-3:]):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret

    def _format_legal_actions_prompt(self, obs: EnvState) -> str:
        if hasattr(self.env, "format_legal_actions_prompt"):
            return self.env.format_legal_actions_prompt(obs)
        return ""

    def _get_legal_actions(self, obs: EnvState) -> Dict[str, List[str]]:
        if hasattr(self.env, "get_legal_actions"):
            return self.env.get_legal_actions(obs)
        return {}

    def _get_recommended_response(self, obs: EnvState) -> Optional[str]:
        if not hasattr(self.env, "get_recommended_plan"):
            return None
        plan = self.env.get_recommended_plan(obs)
        if not plan:
            return None
        lines = ["EXECUTE"]
        for agent_name in self.robot_agent_names:
            if agent_name not in plan:
                return None
            lines.append(f"NAME {agent_name} ACTION {plan[agent_name]}")
        return "\n".join(lines)

    def _response_from_actions(self, actions: Dict[str, str]) -> str:
        lines = ["EXECUTE"]
        for agent_name in self.robot_agent_names:
            lines.append(f"NAME {agent_name} ACTION {actions.get(agent_name, 'WAIT')}")
        return "\n".join(lines)

    def _extract_action_lines(self, response: str) -> Dict[str, str]:
        if not response or "EXECUTE" not in response:
            return {}
        execute_str = response.split("EXECUTE", 1)[1]
        actions = {}
        for raw_line in execute_str.splitlines():
            line = raw_line.strip()
            if not line or "NAME" not in line or "ACTION" not in line:
                continue
            agent_name = line.split("NAME", 1)[1].split("ACTION", 1)[0].strip()
            action = line.split("ACTION", 1)[1].strip()
            actions[agent_name] = action
        return actions

    def _validate_against_legal_actions(
        self,
        obs: EnvState,
        response: str,
        legal_actions: Dict[str, List[str]],
        forbidden_actions: Dict[str, set],
    ) -> Tuple[bool, str]:
        """Generic verification layer before parser/RRT.

        This layer is intentionally task-agnostic where possible, and then
        delegates stronger task-specific checks to optional env hooks:
        - get_allowed_action_names()
        - get_max_parallel_actions(obs)
        - verify_plan_semantics(obs, actions)

        If an env also exposes get_legal_actions(obs), actions must be exact
        members of that list.  Envs without legal-action lists still benefit
        from format/action-name/max-parallel/duplicate/forbidden checks plus
        their task-specific hook.
        """
        actions = self._extract_action_lines(response)
        expected_agents = list(self.robot_agent_names)
        missing = [agent for agent in expected_agents if agent not in actions]
        extra = [agent for agent in actions if agent not in expected_agents]
        if missing or extra:
            return False, f"Plan must contain exactly one action for each robot. missing={missing}, extra={extra}"

        allowed_action_names = None
        if hasattr(self.env, "get_allowed_action_names"):
            allowed_action_names = set(self.env.get_allowed_action_names())

        picked_objects = []
        placed_targets = []
        active_actions = []
        for agent_name, action in actions.items():
            first_token = action.split()[0] if action.split() else ""
            if allowed_action_names is not None and first_token not in allowed_action_names:
                return False, (
                    f"Invalid action name for {agent_name}: '{first_token}'. "
                    f"Allowed action names: {sorted(allowed_action_names)}"
                )

            legal_for_agent = legal_actions.get(agent_name, []) if legal_actions else []
            if legal_actions and action not in legal_for_agent:
                return False, (
                    f"Illegal action for {agent_name}: '{action}'. "
                    f"Choose one of: {legal_for_agent}"
                )
            if action in forbidden_actions.get(agent_name, set()):
                return False, f"Action for {agent_name} repeats a failed action this round: '{action}'"
            if action != "WAIT":
                active_actions.append((agent_name, action))
            if "PICK" in action and "PLACE" in action:
                obj = action.split("PICK", 1)[1].split("PLACE", 1)[0].strip()
                target = action.split("PLACE", 1)[1].strip()
                picked_objects.append(obj)
                placed_targets.append(target)
            elif action.startswith("PICK "):
                obj = action.split("PICK", 1)[1].split("PATH", 1)[0].strip().split()
                if obj:
                    picked_objects.append(obj[0])
            elif action.startswith("PLACE "):
                parts = action.split("PLACE", 1)[1].split("PATH", 1)[0].strip().split()
                if len(parts) >= 2:
                    placed_targets.append(parts[1])
            elif action.startswith("PUT "):
                parts = action.split("PUT", 1)[1].split("PATH", 1)[0].strip().split()
                if len(parts) >= 2:
                    placed_targets.append(parts[1])
        max_parallel_actions = None
        if hasattr(self.env, "get_max_parallel_actions"):
            try:
                max_parallel_actions = self.env.get_max_parallel_actions(obs)
            except TypeError:
                max_parallel_actions = self.env.get_max_parallel_actions()
        if max_parallel_actions is not None and len(active_actions) > max_parallel_actions:
            return False, (
                f"Too many non-WAIT actions: {len(active_actions)}. "
                f"This task allows at most {max_parallel_actions} non-WAIT action(s) per round. "
                f"Active actions: {active_actions}. Use WAIT for the other robots."
            )
        duplicate_objects = sorted({obj for obj in picked_objects if picked_objects.count(obj) > 1})
        if duplicate_objects:
            return False, f"Multiple robots cannot PICK the same object in one round: {duplicate_objects}"
        duplicate_targets = sorted({target for target in placed_targets if placed_targets.count(target) > 1})
        if duplicate_targets:
            return False, f"Multiple robots should not PLACE into the same target in one round: {duplicate_targets}"

        if hasattr(self.env, "verify_plan_semantics"):
            valid, reason = self.env.verify_plan_semantics(obs, actions)
            if not valid:
                return False, f"Task semantic verification failed: {reason}"
        return True, "OK"

    def _extract_agents_from_text(self, text: str) -> List[str]:
        failed_agents = []
        if not text:
            return failed_agents
        for agent_name in self.robot_agent_names:
            patterns = [
                f"Action for {agent_name}",
                f"Illegal action for {agent_name}",
                f"{agent_name}'s ACTION",
                f"Out of reach: {agent_name}",
                f"IK failed: on {agent_name}",
            ]
            if any(pattern in text for pattern in patterns):
                failed_agents.append(agent_name)
        return failed_agents

    def _ban_actions_from_response(
        self,
        response: str,
        forbidden_actions: Dict[str, set],
        agents: Optional[List[str]] = None,
    ) -> None:
        agents_to_ban = set(agents) if agents else None
        for agent_name, action in self._extract_action_lines(response).items():
            if agents_to_ban is not None and agent_name not in agents_to_ban:
                continue
            if action != "WAIT":
                forbidden_actions.setdefault(agent_name, set()).add(action)

    def _format_forbidden_actions(self, forbidden_actions: Dict[str, set]) -> str:
        if not any(forbidden_actions.values()):
            return ""
        lines = ["[Forbidden Actions This Replan Round]"]
        for agent_name in self.robot_agent_names:
            for action in sorted(forbidden_actions.get(agent_name, [])):
                lines.append(f"- {agent_name}: {action}")
        lines.append("Do not repeat forbidden actions; choose another listed legal action or WAIT.")
        return "\n".join(lines) + "\n"

    def _should_ban_individual_actions(self, feedback: str) -> bool:
        """Whether a failed multi-robot plan invalidates each action alone.

        Reachability, IK, parsing, and illegal-action failures usually point to
        a specific robot/action.  A collision between two robots, however, often
        means the *combination* is bad while each single action is still useful.
        Do not ban individual actions on collision feedback; fallback can then
        try executable subsets such as "Bob only" or "Chad only".
        """
        return "Collision detected" not in feedback

    def _make_feedback_more_actionable(self, feedback: str) -> str:
        if "Collision detected" not in feedback:
            return feedback
        return (
            feedback
            + "\nCollision feedback means the concurrent robot combination is unsafe. "
            + "Try fewer simultaneous actions, preferably one active robot and others WAIT."
        )

    def _try_parse_and_feedback(self, obs: EnvState, response: str):
        parse_succ, parsed_str, plans = self.parser.parse(obs, response)
        if not parse_succ:
            return False, f"Parsing failed: {parsed_str}", []
        for plan in plans:
            ready, feedback = self.feedback_manager.give_feedback(plan)
            if not ready:
                return False, feedback, []
        return True, "None", plans

    def _partial_fallback_responses(
        self,
        obs: EnvState,
        legal_actions: Dict[str, List[str]],
        forbidden_actions: Dict[str, set],
    ) -> List[str]:
        responses = []
        seen = set()

        def add_if_valid(actions: Dict[str, str]) -> None:
            response = self._response_from_actions(actions)
            if response in seen:
                return
            seen.add(response)
            valid, _ = self._validate_against_legal_actions(
                obs,
                response,
                legal_actions,
                forbidden_actions,
            )
            if valid:
                responses.append(response)

        # 1) Start from the env's recommended plan and try smaller executable
        # subsets.  This avoids one bad robot action blocking useful moves by
        # other robots.
        fallback = self._get_recommended_response(obs)
        if fallback:
            base_actions = self._extract_action_lines(fallback)
            active_agents = [
                agent for agent in self.robot_agent_names
                if base_actions.get(agent, "WAIT") != "WAIT"
            ]
            for size in range(len(active_agents), 0, -1):
                from itertools import combinations
                for combo in combinations(active_agents, size):
                    actions = {agent: "WAIT" for agent in self.robot_agent_names}
                    for agent in combo:
                        actions[agent] = base_actions[agent]
                    add_if_valid(actions)

        # 2) If all recommended subsets are blocked by IK/forbidden-action
        # feedback, enumerate other legal one- or two-robot actions.  Keep this
        # conservative: no duplicate picked object and no duplicate place target.
        atomic_candidates = []
        for agent_name in self.robot_agent_names:
            for action in legal_actions.get(agent_name, []):
                if action == "WAIT" or action in forbidden_actions.get(agent_name, set()):
                    continue
                atomic_candidates.append((agent_name, action))

        def compatible(combo: Tuple[Tuple[str, str], ...]) -> bool:
            agents = [agent for agent, _ in combo]
            if len(set(agents)) != len(agents):
                return False
            picked = []
            targets = []
            for _, action in combo:
                if "PICK" in action and "PLACE" in action:
                    picked.append(action.split("PICK", 1)[1].split("PLACE", 1)[0].strip())
                    targets.append(action.split("PLACE", 1)[1].strip())
            return len(set(picked)) == len(picked) and len(set(targets)) == len(targets)

        from itertools import combinations
        max_parallel_actions = min(2, len(self.robot_agent_names))
        if hasattr(self.env, "get_max_parallel_actions"):
            try:
                max_parallel_actions = min(max_parallel_actions, self.env.get_max_parallel_actions(obs))
            except TypeError:
                max_parallel_actions = min(max_parallel_actions, self.env.get_max_parallel_actions())
        for size in range(max_parallel_actions, 0, -1):
            for combo in combinations(atomic_candidates, size):
                if not compatible(combo):
                    continue
                actions = {agent: "WAIT" for agent in self.robot_agent_names}
                for agent, action in combo:
                    actions[agent] = action
                add_if_valid(actions)
        return responses

    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [],
        obs: Optional[EnvState] = None,
        forbidden_actions: Optional[Dict[str, set]] = None,
        ):

        task_desp = self.env.describe_task_context() # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n"
        if obs is not None and hasattr(self.env, "get_plan_state_prompt"):
            full_prompt += self.env.get_plan_state_prompt(obs) + "\n"
        if obs is not None:
            legal_actions_prompt = self._format_legal_actions_prompt(obs)
            if legal_actions_prompt:
                full_prompt += legal_actions_prompt + "\n"
        if forbidden_actions is not None:
            full_prompt += self._format_forbidden_actions(forbidden_actions)

        if self.use_history:
            history_desp = self.compose_round_history()
            full_prompt += history_desp + "\n"

        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans)
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt

        if self.comm_mode == "plan":
            comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env)
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt

    def prompt_one_round(self, obs: EnvState, save_path: str = ""):
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        ready_to_execute = False
        llm_plans = []
        forbidden_actions = {}
        legal_actions = self._get_legal_actions(obs)
        for i in range(self.num_replans):
            system_prompt = self.compose_system_prompt(
                obs_desp,
                plan_feedbacks,
                obs=obs,
                forbidden_actions=forbidden_actions,
            )
            response, usage = self.query_once(
                system_prompt, user_prompt=""
                ) # NOTE: single_thread doesn't use user role
            response_history.append(response)

            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

            curr_feedback = "None"
            valid_legal, legal_reason = self._validate_against_legal_actions(
                obs,
                response,
                legal_actions,
                forbidden_actions,
            )
            if not valid_legal:
                curr_feedback = f"""
Action candidate validation failed! {legal_reason}
Previous response:
{response}
Choose exactly one action per robot from [Legal Actions]. Do not invent actions.
                """
                failed_agents = self._extract_agents_from_text(legal_reason)
                # A max-parallelism violation means the combination is unsafe,
                # not that any individual action is bad.  Do not ban the
                # single actions; fallback/replan can then try one of them with
                # other robots WAITing.
                if "Too many non-WAIT actions" not in legal_reason:
                    self._ban_actions_from_response(
                        response,
                        forbidden_actions,
                        agents=(failed_agents or None),
                    )
                ready_to_execute = False
                parse_succ = False
                llm_plans = []
            else:
            # try parsing
                parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response)
                if not parse_succ:
                    execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                    curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                    """
                    failed_agents = self._extract_agents_from_text(parsed_str)
                    self._ban_actions_from_response(
                        response,
                        forbidden_actions,
                        agents=(failed_agents or None),
                    )
                    ready_to_execute = False
            # give env. feedback
                else:
                    ready_to_execute = True
                    for j, llm_plan in enumerate(llm_plans):
                        ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                        if not ready_to_execute:
                            curr_feedback = self._make_feedback_more_actionable(env_feedback)
                            if self._should_ban_individual_actions(env_feedback):
                                failed_agents = self._extract_agents_from_text(env_feedback)
                                self._ban_actions_from_response(
                                    response,
                                    forbidden_actions,
                                    agents=(failed_agents or None),
                                )
                            break

            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

            if ready_to_execute:
                plan_str = parsed_str
                break

        if not ready_to_execute:
            fallback_attempts = []
            for fallback_response in self._partial_fallback_responses(obs, legal_actions, forbidden_actions):
                fallback_ready, fallback_feedback, fallback_plans = self._try_parse_and_feedback(
                    obs,
                    fallback_response,
                )
                fallback_attempts.append(
                    {
                        "response": fallback_response,
                        "feedback": fallback_feedback,
                        "ready": fallback_ready,
                    }
                )
                if fallback_ready:
                    ready_to_execute = True
                    llm_plans = fallback_plans
                    response_history.append(fallback_response)
                    break
            if fallback_attempts:
                timestamp = datetime.now().strftime("%m%d-%H%M")
                json.dump(
                    [
                        {
                            "sender": "Planner",
                            "message": fallback_attempts[-1]["response"],
                        },
                        {
                            "sender": "Feedback",
                            "message": (
                                "Used deterministic partial fallback after LLM replans failed."
                                if ready_to_execute
                                else "Deterministic partial fallback attempted but no candidate passed feedback."
                            ),
                        },
                        {
                            "sender": "FallbackAttempts",
                            "message": json.dumps(fallback_attempts, indent=2),
                        },
                    ],
                    open(f"{save_path}/fallback_{timestamp}.json", "w"),
                )
        self.response_history = response_history
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                response, usage = query_ollama_chat(
                    model=self.llm_source,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as e:
                print(f"API error, try again: {e}")
                response = ""
                usage = {"error": str(e), "model": self.llm_source}
            continue
        return response, usage



    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success:
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            self.round_history.append(
                f"[Executed Action]\n{parsed_plan}\n"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = []
        self.response_history = []
