import os
import copy
import time
import cv2 
import random
import numpy as np  
from pydantic import dataclasses, validator 
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control 
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS,PANDA_CONSTANTS

ROPE_FRONT_BODY="CB0"
ROPE_BACK_BODY="CB24"
ROPE_TASK_OBJECTS=[  
    "rope",
    "table_top",
    "obstacle_wall",
    "groove_left_end",
    "groove_right_end",
]
ROPE_INIT_RANGE = (
    np.array([-1.2, 0.6, 0.2]),
    np.array([-1.3, 0.45, 0.2]),
)
OBSTACLE_RANGE = (
    np.array([-0.15, 0.5, 0.2]),
    np.array([0, 0.5, 0.3]),
)


OBSTACLE_CORNER_NAMES = [
    "obstacle_wall_front_top",
    # "obstacle_wall_front_bottom",
    "obstacle_wall_back_top",
    # "obstacle_wall_back_bottom",
]
ROPE_TASK_CONTEXT="""2 robots, Alice and Bob, together must pick up a long rope and put it precisely into a narrow groove slot. They must lift up the rope together, each grasping one side of the rope. 
There's an obstacle block wall between the rope and the groove, so the robots must lift the rope **high** above the obstacle block top before putting it into the groove.
The rope is heavy, so they must hold the rope at the same time, and distance between their grippers must stay fixed, so the rope doesn't drop.
At each round, if given 'Scene description' and 'Environment feedback', use it to reason about the task and improve any previous plans. 
Each robot should reach for the closet target, if a robot failed IK to reach a goal, change strategy to a different action.
"""

ROPE_ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty, <object> can be either rope_front_end or rope_back_end
2) PUT <obj> <location> PATH <path>: PUT on either groove_left_end or groove_right_end only if you have already PICKed the rope, each end can only hold only one end of the rope, not both.
3) WAIT PATH <path>: do nothing. Because this task uses action_and_path mode, even WAIT must include exactly four coordinates; use four copies of the current gripper location.
Valid ACTION names are only PICK, PUT, and WAIT. Never output LIFT, MOVE, PLACE, LOWER, RAISE, or DROP.
To lift or move the rope while holding it, use PUT <obj> <location> PATH <path> with high intermediate PATH coordinates; do not use a LIFT action.
Choose a different <obj> to PICK or PUT if Environment Feedback failed.
Every robot line must contain PATH. Each <path> must contain exactly four coordinates, each must be evenly distanced from each other and interpolates between start and goal.
PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
The PATHs must do top-down pick or place: 
- move directly atop the rope's end by height 0.2 before PICK: e.g. Alice's gripper is at (0, 0, 0.3), rope_front_end is at (-0.25, 0.39, 0.29): ACTION PICK rope_front_end PATH [(0, 0.1, 0.3),(0, 0.2, 0.49),(-0.1, 0.25, 0.49),(-0.25, 0.39, 0.49)]
- lift rope up before moving it to PUT: e.g. Bob's gripper is at (0.9, 0, 0.2), groove_left_end is at (0.35, 0.35, 0.43): ACTION PUT rope_front_end groove_left_end PATH [(0.9,0.0,0.50), (0.5, 0, 0.52), (0.2, 0.1, 0.52),(0.35, 0.35, 0.50)]
Keep PATH z coordinates above the wall top but below the stated maximum height. Usually use z around 0.50 to 0.54 after the rope is picked; do not use z=0.8.

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PUT rope_front_end groove_right_end PATH <path>\nNAME Bob ACTION PUT rope_back_end groove_left_end PATH <path>\n'
"""

ROPE_TASK_CHAT_PROMPT="""They discuss to find the best paths. Carefully consider environment feedback and others' responses. Robots must coordinate paths to avoid collision.
They talk in order [Alice],[Bob],[Alice],..., after reaching agreement, plan exactly one ACTION per robot, output an EXECUTE to summarize the plan, and stop talking.
Their chat and final plan are: """

ROPE_TASK_PLAN_PROMPT="""Plan one action for each robot. Analyze whether each robot is empty-handed or already holding a rope end.
Use only valid rope actions: PICK, PUT, or WAIT. Never output LIFT or PLACE.
Strict stage rule:
1) First make sure both rope ends are held: Alice must PICK rope_front_end and Bob must PICK rope_back_end.
2) Only after both robots are holding rope ends, both robots PUT the rope ends into the groove in the same round.
If exactly one robot is holding a rope end, that holding robot must WAIT while the empty-handed robot PICKs the other rope end. Do not PUT with only one rope end held.
If both robots are already holding rope ends, the next action should usually be PUT for both robots, with PATHs that lift the rope above the wall and move it toward groove_left_end/groove_right_end.
Never mix PUT and PICK in the same round.
Keep both grippers at similar heights and preserve the rope-end distance as much as possible.
Plan PATHs that efficiently achieve the task, avoid collision, and stay within the stated height limits.
Prefer [Recommended Rope Stage Plan] exactly unless environment feedback says it failed.
Output only the final EXECUTE block:
"""

class MoveRopeTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_rope.xml",
        one_obj_each: bool = False,
        **kwargs,
    ):    
        self.robot_names = ["ur5e_robotiq", "panda"] 
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob", 
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda", 
        }
        self.robots = dict()  

        super(MoveRopeTask, self).__init__(
            filepath=filepath, 
            task_objects=ROPE_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=UR5E_ROBOTIQ_CONSTANTS,
                panda=PANDA_CONSTANTS,
            ),
            **kwargs
        ) 
        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()
        # robotiq_config["ik_joint_names"].remove("ur5e_0_base_joint")
        # robotiq_config["all_joint_names"].remove("ur5e_0_base_joint")
        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )
        panda_config = PANDA_CONSTANTS.copy()
        # panda_config["ik_joint_names"].remove("panda_base_joint")
        # panda_config["all_joint_names"].remove("panda_base_joint")
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )
         
        self.align_threshold = 0.2
        self.groove_pos = dict()
        for side in ["left", "right"]:
            self.groove_pos[f"groove_{side}_end"] = self.physics.data.site(f"groove_{side}_end").xpos.copy()
        self.rope_length = np.linalg.norm(
            self.physics.data.body(ROPE_FRONT_BODY).xpos - self.physics.data.body(ROPE_BACK_BODY).xpos
            )
        
    @property
    def waypoint_std_threshold(self):
        return 0.3

    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]: 
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]
        if 'groove' in target_name:
            sname = "groove_left_end" if 'left' in target_name else "groove_right_end"
            ret = self.physics.data.site(sname).xpos.copy() 
        
        elif target_name in ["obstacle_wall_front_top", "obstacle_wall_back_top"]:
            ret = self.physics.data.site(target_name).xpos.copy()
            ret[2] += 0.1
        
        elif target_name == "rope_front_end":
            ret = self.physics.data.body(ROPE_FRONT_BODY).xpos.copy()
            # The rope body can lie very close to the tabletop.  A gripper goal
            # only 0.1m above the rope often collides with the table at the
            # PICK goal step, especially for the robotiq hand.  Grasping the
            # rope in this task is implemented by activating the rope-end weld,
            # so a slightly higher gripper target is safer while still picking
            # the correct end.
            ret[2] = np.clip(ret[2] + 0.35, 0.42, 0.54)
        elif target_name == "rope_back_end":
            ret = self.physics.data.body(ROPE_BACK_BODY).xpos.copy()
            ret[2] = np.clip(ret[2] + 0.35, 0.42, 0.54)

        return ret 
         
    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if 'groove' in target_name or target_name in ["obstacle_wall_front_top", "obstacle_wall_back_top"]:
            sname = target_name 

        elif target_name in ["rope_front_end", "rope_back_end"]:
            body_name = ROPE_FRONT_BODY if target_name == "rope_front_end" else ROPE_BACK_BODY
            return self.physics.data.body(body_name).xquat.copy() 

        else:
            return None

        xmat = self.physics.data.site(target_name).xmat.copy()
        ret = mat_to_quat(xmat.reshape(3,3))
        return ret

    def _agent_holding_rope(self, obs: EnvState, agent_name: str) -> Optional[str]:
        robot_name = self.robot_name_map_inv[agent_name]
        contacts = getattr(obs, robot_name).contacts
        if ROPE_FRONT_BODY in contacts:
            return "rope_front_end"
        if ROPE_BACK_BODY in contacts:
            return "rope_back_end"
        return None

    def _agent_ee_pos(self, obs: EnvState, agent_name: str) -> np.ndarray:
        robot_name = self.robot_name_map_inv[agent_name]
        return getattr(obs, robot_name).ee_xpos.copy()

    def _format_path(self, pts: List[np.ndarray]) -> str:
        return "[" + ", ".join(
            f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in pts
        ) + "]"

    def _wait_action(self, obs: EnvState, agent_name: str) -> str:
        pos = self._agent_ee_pos(obs, agent_name)
        return f"WAIT PATH {self._format_path([pos, pos, pos, pos])}"

    def _pick_action(self, obs: EnvState, agent_name: str, rope_end: str) -> str:
        start = self._agent_ee_pos(obs, agent_name)
        target = self.get_target_pos(agent_name, rope_end).copy()
        safe_z = min(0.54, max(0.50, target[2]))
        pts = [
            np.array([start[0], start[1], min(0.54, max(0.42, start[2]))]),
            np.array([start[0], (start[1] + target[1]) / 2, safe_z]),
            np.array([(start[0] + target[0]) / 2, target[1], safe_z]),
            np.array([target[0], target[1], safe_z]),
        ]
        return f"PICK {rope_end} PATH {self._format_path(pts)}"

    def _put_action(self, obs: EnvState, agent_name: str, rope_end: str, groove_end: str) -> str:
        start = self._agent_ee_pos(obs, agent_name)
        target = self.get_target_pos(agent_name, groove_end).copy()
        safe_z = 0.54
        pts = [
            np.array([start[0], start[1], min(0.54, max(0.42, start[2] + 0.10))]),
            np.array([(start[0] * 2 + target[0]) / 3, (start[1] * 2 + target[1]) / 3, safe_z]),
            np.array([(start[0] + target[0] * 2) / 3, (start[1] + target[1] * 2) / 3, safe_z]),
            np.array([target[0], target[1], 0.50]),
        ]
        return f"PUT {rope_end} {groove_end} PATH {self._format_path(pts)}"

    def get_recommended_plan(self, obs: EnvState) -> Dict[str, str]:
        """Deterministic rope stage policy: first both PICK, then both PUT."""
        holding = {
            agent_name: self._agent_holding_rope(obs, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        plan = {
            "Alice": self._wait_action(obs, "Alice"),
            "Bob": self._wait_action(obs, "Bob"),
        }

        if holding["Alice"] is None and holding["Bob"] is None:
            plan["Alice"] = self._pick_action(obs, "Alice", "rope_front_end")
            plan["Bob"] = self._pick_action(obs, "Bob", "rope_back_end")
        elif holding["Alice"] is not None and holding["Bob"] is None:
            plan["Alice"] = self._wait_action(obs, "Alice")
            plan["Bob"] = self._pick_action(obs, "Bob", "rope_back_end")
        elif holding["Alice"] is None and holding["Bob"] is not None:
            plan["Alice"] = self._pick_action(obs, "Alice", "rope_front_end")
            plan["Bob"] = self._wait_action(obs, "Bob")
        else:
            # Reward accepts either orientation, but front->left and back->right
            # is the shortest and matches successful traces in this workspace.
            plan["Alice"] = self._put_action(obs, "Alice", holding["Alice"], "groove_left_end")
            plan["Bob"] = self._put_action(obs, "Bob", holding["Bob"], "groove_right_end")
        return plan

    def format_legal_actions_prompt(self, obs: EnvState) -> str:
        plan = self.get_recommended_plan(obs)
        holding = {
            agent_name: self._agent_holding_rope(obs, agent_name) or "nothing"
            for agent_name in ["Alice", "Bob"]
        }
        lines = [
            "[Rope Stage State]",
            f"- Alice holding: {holding['Alice']}",
            f"- Bob holding: {holding['Bob']}",
            "[Recommended Rope Stage Plan]",
            "Follow this staged plan exactly unless environment feedback says it failed:",
            "EXECUTE",
            f"NAME Alice ACTION {plan['Alice']}",
            f"NAME Bob ACTION {plan['Bob']}",
        ]
        return "\n".join(lines) + "\n"

    def get_graspable_objects(self):
        graspables = [
            # "rope_front_end",
            # "rope_back_end",
            "groove_left_end",
        ]
        return dict(
            Alice=graspables,
            Bob=graspables, 
        )

    def get_grasp_site(self, obj_name: str = "rope_end") -> Optional[str]:

        if obj_name in ["rope_front_end", "rope_back_end"]:
            return f"{obj_name}"
        
        if obj_name in ["rope_front", "rope_back"]:
            return f"{obj_name}_end"
        
        if 'groove_left' in obj_name:
            return "groove_left_end"
        
        if 'groove_right' in obj_name:
            return "groove_right_end"
        
        else:
            return None
    
    def get_reward_done(self, obs): 
        # task specific!
        rew = 0
        done = False 
        groove_left = self.physics.data.site("groove_left_end").xpos.copy()[:2]
        groove_right = self.physics.data.site("groove_right_end").xpos.copy()[:2]
        rope_front = self.physics.data.body(ROPE_FRONT_BODY).xpos.copy()[:2]
        rope_back = self.physics.data.body(ROPE_BACK_BODY).xpos.copy()[:2]

        dist_lf = np.linalg.norm(groove_left - rope_front)
        dist_lb = np.linalg.norm(groove_left - rope_back)
        dist_rf = np.linalg.norm(groove_right - rope_front)
        dist_rb = np.linalg.norm(groove_right - rope_back)
        
        touch_bottom = 'groove_bottom' in obs.objects['rope'].contacts

        if (dist_lf < self.align_threshold and dist_rb < self.align_threshold) or \
            (dist_lb < self.align_threshold and dist_rf < self.align_threshold):
            done = True and touch_bottom
            rew = 1 if done else 0
        return rew, done

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]
    
    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs
    
    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.4, 1.6), y=(-0.4, 1.5), z=(0, 1))
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-1.3, 1.6), y=(0, 1.5), z=(0, 1))
        else:
            raise NotImplementedError
    
    def sample_initial_scene(self): 
        # sample locations of the cabinet
        low, high = ROPE_INIT_RANGE
        new_pos = self.random_state.uniform(low, high) 
        new_angle = self.random_state.uniform(low=-np.pi/6, high=np.pi/6)
        new_quat = Quaternion(
            axis=[0,0,1], angle=new_angle
            ) 
        new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
        if abs(new_angle) > 0.3:
            new_pos[0] = -1.1
        self.reset_body_pose(
            body_name="rope",
            pos=new_pos,
            # quat=new_quat,
        )  
        self.reset_qpos(
            jnt_name="rope_joint",
            pos=new_pos,
            quat=new_quat,  
        )
        # apply random force to init the rope: 
        rope_body = self.random_state.choice(range(8, 16))
        rope_body = f'CB{rope_body}'
        self.physics.named.data.xfrc_applied[[rope_body], 2] = self.random_state.uniform(1, 1.4, size=1).reshape((1, 1))
        wall_pos = self.random_state.uniform(low=OBSTACLE_RANGE[0], high=OBSTACLE_RANGE[1])
        new_quat = Quaternion(
            axis=[0,0,1], angle=self.random_state.uniform(low=-np.pi/5, high=np.pi/5)
            ) 
        new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
        
        self.reset_body_pose(
            body_name="obstacle_wall",
            pos=wall_pos,
            quat=new_quat, # no resample
        )
             
        self.physics.forward()
        for i in range(3):
            self.physics.step(50) 

        self.physics.named.data.xfrc_applied[[rope_body], 2] = 0
        self.physics.forward()
        for i in range(6):
            self.physics.step(50) 

        
        self.physics.forward()
        self.physics.step(50)
    
    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        rope_ids = self.get_all_body_ids("rope")
        table_id = self.physics.model.body("table").id
        groove_ids = self.get_all_body_ids("groove")
        ret = []

        for rope_id in rope_ids:
            ret.append((table_id, rope_id))
            for groove_id in groove_ids:
                ret.append((groove_id, rope_id))

        for link_id in self.robots["Alice"].all_link_body_ids + self.robots["Bob"].all_link_body_ids:
            for rope_id in rope_ids:
                ret.append((link_id, rope_id)) 

        return ret
    
    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs 
    
    def describe_robot_state(self, obs, robot_name: str = "panda"):
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        contacts = robot_state.contacts 
        if len(contacts) == 0:
            obj = "nothing"
        else:
            obj = ",".join([c for c in contacts]) 
            if ROPE_FRONT_BODY in obj:
                obj = 'rope_front_end'
            elif ROPE_BACK_BODY in obj:
                obj = 'rope_back_end'
        agent_name = self.robot_name_map[robot_name]
        robot_desp = f"{agent_name}'s gripper: ({x:.2f}, {y:.2f}, {z:.2f}), holding {obj}"
        return robot_desp  
 
    def describe_obs(self, obs: EnvState):
        object_desp =  "[Scene description]\n"
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        object_desp += f"robots must move lower than 0.55 but higher than table height {table_height:.2f}\n"
        for side in ["left", "right"]:
            sname = f"groove_{side}_end"
            x,y,z = self.groove_pos[sname]
            object_desp += f"{sname}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
       
        for name, body_name in zip(
            ["rope_front_end", "rope_back_end"],
            [ROPE_FRONT_BODY, ROPE_BACK_BODY],
        ):
            x,y,z = self.physics.data.body(body_name).xpos
            object_desp += f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
        
        object_desp += "The obstacle wall location: "
        for name in OBSTACLE_CORNER_NAMES:
            x,y,z = self.physics.data.site(name).xpos 
            object_desp += f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"

        robot_desp = "The robots:\n"
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name) + "\n"
            
        full_desp = object_desp + robot_desp
        return full_desp 
    
    def get_task_feedback(self, llm_plan, pose_dict):
        """Get the feedback on planned target poses for each robot at the same time step"""
        task_feedback = ""
        obs = self.get_obs() 
        # if len(obs.panda.contacts) > 0 and len(obs.ur5e_robotiq.contacts) > 0:
        #     # if ("rope_front_end" in obs.panda.contacts and "rope_back_end" in obs.ur5e_robotiq.contacts) or \
        #     #     ("rope_front_end" in obs.ur5e_robotiq.contacts and "rope_back_end" in obs.panda.contacts):
        #     pose1 = pose_dict["Alice"]
        #     pose2 = pose_dict["Bob"]
        #     dist = np.linalg.norm(pose1.position - pose2.position)
        #     if dist < self.rope_length * 0.8 or dist > self.rope_length * 1.2:
        #         task_feedback += f"PATH at: Alice {pose1.pos_string}, Bob: {pose2.pos_string} are wrong: distance between them is {dist}, but it must be {self.rope_length:.2f}"   
        for agent_name, action_str in llm_plan.action_strs.items(): 
            # WAIT is parsed and useful as a conservative fallback when one
            # robot's simultaneous rope action is causing IK/collision failure.
            if 'PLACE' in action_str or 'MOVE' in action_str or 'LIFT' in action_str:
                task_feedback += f"{agent_name}'s ACTION is not supported" 

        actions = llm_plan.action_strs
        has_put = any('PUT' in action for action in actions.values())
        has_pick = any('PICK' in action for action in actions.values())
        if has_put and has_pick:
            task_feedback += (
                "Do not mix PICK and PUT in the same round. "
                "If only one robot is holding a rope end, the holding robot should WAIT "
                "while the other robot PICKs the other end.\n"
            )
        if has_put and not all('PUT' in action for action in actions.values()):
            task_feedback += (
                "The rope must be PUT by both robots at the same time. "
                "Do not PUT with only one robot; use WAIT for the holding robot until both ends are held.\n"
            )

        holding = {
            agent_name: self._agent_holding_rope(obs, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        num_holding = sum(v is not None for v in holding.values())

        if num_holding == 0:
            if not (
                "PICK rope_front_end" in actions.get("Alice", "")
                and "PICK rope_back_end" in actions.get("Bob", "")
            ):
                task_feedback += (
                    "Stage rule violated: when no robot is holding the rope, "
                    "Alice must PICK rope_front_end and Bob must PICK rope_back_end in the same round. "
                    "Do not WAIT or PUT before both ends are picked.\n"
                )
        elif num_holding == 1:
            holder = [agent for agent, held_end in holding.items() if held_end is not None][0]
            empty = "Bob" if holder == "Alice" else "Alice"
            if 'PUT' in actions.get(holder, ''):
                task_feedback += (
                    f"{holder} is the only robot holding the rope. {holder} should WAIT, "
                    "and the other robot should PICK the other rope end before any PUT.\n"
                )
            if 'WAIT' not in actions.get(holder, ''):
                task_feedback += (
                    f"{holder} is already holding {holding[holder]}; {holder} must WAIT until both ends are held.\n"
                )
            expected_pick = "rope_back_end" if empty == "Bob" else "rope_front_end"
            if f"PICK {expected_pick}" not in actions.get(empty, ""):
                task_feedback += (
                    f"{empty} is empty-handed and must PICK {expected_pick} before any PUT.\n"
                )
        else:
            if not all('PUT' in action for action in actions.values()):
                task_feedback += (
                    "Stage rule violated: both robots are holding rope ends, so both robots should PUT their ends into the groove in the same round.\n"
                )
        return task_feedback
    
    def get_agent_prompt(self, obs: EnvState, agent_name: str):
        robot_name = self.robot_name_map_inv[agent_name]
        other_robot = "Alice" if agent_name == "Bob" else "Bob"
 
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        obj_desp = []
        for side in ["left", "right"]:
            sname = f"groove_{side}_end"
            x,y,z = self.groove_pos[sname]
            obj_desp.append(f"{sname}: ({x:.2f}, {y:.2f}, {z:.2f})")
       
        for name, body_name in zip(
            ["rope_front_end", "rope_back_end"],
            [ROPE_FRONT_BODY, ROPE_BACK_BODY],
        ):
            x,y,z = self.physics.data.body(body_name).xpos
            obj_desp.append(f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        for name in OBSTACLE_CORNER_NAMES:
            x,y,z = self.physics.data.site(name).xpos 
            obj_desp.append(f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        obj_desp = '\n'.join(obj_desp)
        robot_desp = self.describe_robot_state(obs, robot_name).replace(f"{agent_name}'s", "Your")            

        agent_prompt = f"""
You are {agent_name}, together with {other_robot} you must pick up a long rope and put it precisely into a narrow groove slot. 
You two must lift up the rope together, each grasping one side of the rope. 
There's an obstacle block wall between rope and groove, so both of you must lift the rope **high** above the obstacle block top before putting it into the groove.
The rope is heavy, so you must always pick and put the rope at the same time.
Distance between your grippers must stay fixed, so the rope doesn't drop.

Your gripper must move lower than 0.55 but higher than table height {table_height:.2f}
At the current round: 
{robot_desp}
Object locations: 
{obj_desp}

Never forget you are {agent_name}!
Think step-by-step about the task and {other_robot}'s response. Check and correct {other_robot} if they made a mistake. 
Discuss with {other_robot} to come up with the best plan and smooth, collision-free paths, propose exactly one action for this round. 
Use [Environment Feedback] to improve your actions and waypoint paths. Reach for the closest target, if any waypoint failed IK, change strategy to a different action.

When you respond, tell {other_robot} about your status. Respond very concisely but informatively, and do not repeat what others have said.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction] and [Path Plan Instruction].
"""
        return agent_prompt

    def describe_robot_capability(self):
        return ""

    def describe_task_context(self):
        context = ROPE_TASK_CONTEXT
        return context

    def get_contact(self):
        
        contacts = super().get_contact()
        # temp fix!  
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if 'CB' in c]
        contacts["panda"] = [c for c in contacts["panda"] if 'CB' in c]
        return contacts

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return ROPE_TASK_CHAT_PROMPT 

    def central_plan_prompt(self):
        return ROPE_TASK_PLAN_PROMPT 

    def get_action_prompt(self) -> str:
        return ROPE_ACTION_SPACE 
        
if __name__ == "__main__":
    env = MoveRopeTask()
    obs = env.reset()  
    print(env.get_agent_prompt(obs, "Alice"))
    print(env.get_agent_prompt(obs, "Bob"))
    breakpoint()
    
