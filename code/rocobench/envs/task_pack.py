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
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, PANDA_CONSTANTS

PACK_TASK_OBJECTS=[
    "bin",
    "table_top",
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_ITEM_NAMES=[
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_BIN_SITE_NAMES=[
    "bin_front_left",
    "bin_front_right",
    "bin_front_middle",
    "bin_back_left",
    "bin_back_right", 
    "bin_back_middle",
]
 
PACK_TASK_CONTEXT="""[Task Description]
Two robots, Alice and Bob, each stands at a different side of the table, and together pack all the grocery items on the table into a bin.
They choose objects closest to their grippers. At each round, they are given [Scene description], [Environment feedback], and must reason about the task. Each robot does **exactly** one ACTION and PATH per round, their PATHs must avoid collision.
"""

PACK_ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty;
2) PLACE <obj> <bin_slot> PATH <path>: only if you have already PICKed the object, you can PLACE it into an empty bin slot, do NOT PLACE if another object is already in a slot!
3) WAIT PATH <path>: do not PICK or PLACE. Because this task uses action_and_path mode, even WAIT must include exactly four coordinates. If another robot is PLACING, WAIT may move/retreat to a safe side pose while keeping any held object; otherwise use four copies of the current gripper location.

Valid bin slots are: bin_front_left, bin_front_right, bin_front_middle, bin_back_left, bin_back_right, bin_back_middle.
Never use the generic word "bin" as a PLACE target; always use one explicit bin slot.
To reduce robot-robot collision near the bin, PICK actions may be parallel, but PLACE actions should be conservative: at most one robot PLACEs per round and the other robot WAITs.

Each <path> must contain exactly four <coord>s that smoothly interpolate between start and goal, coordinates must be evenly distanced from each other.
The robot PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
The PATHs must do top-down pick or place: 
- move directly atop an object by height 0.2 before PICK: e.g. Alice's gripper is at (0, 0, 0.3), banana is at (-0.25, 0.39, 0.29): NAME Alice ACTION PICK banana PATH [(0, 0.1, 0.3),(0, 0.2, 0.49),(-0.1, 0.25, 0.49),(-0.25, 0.39, 0.49)]
- lift an object vertically up before moving it to PLACE: e.g. Bob's gripper is at (0.9, 0, 0.2), bin_front_left is at (0.35, 0.35, 0.43): NAME Bob ACTION PLACE apple bin_front_left PATH [(0.9,0.0,0.5), (0.5, 0, 0.5), (0.2, 0.1, 0.5),(0.35, 0.35, 0.5)]

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PICK apple PATH <path>\nNAME Bob ACTION PLACE banana bin_back_middle PATH <path>\n'
"""

PACK_PLAN_PROMPT="""Plan one action for each robot for Pack Grocery.
Use only valid actions from [Action Options]. Use explicit bin slot names, never use the generic word "bin" as a target.
Follow the conservative collision-avoidance policy:
- If both robots are empty-handed, they may PICK two different unpacked items in parallel, but do not start by picking two tall/central items together; prefer the [Recommended Pack Plan].
- If exactly one robot is holding an item, that robot should PLACE it into an empty bin slot while the other robot WAITs.
- If both robots are holding items, only one robot should PLACE this round and the other robot should WAIT; prefer placing Bob's held item first because Alice placing across the front can collide with Bob's held object.
- If a PLACE failed while the other robot was WAITing but holding an item, switch to PLACE by the other holding robot; WAIT does not remove the held object from collision checking.
- When one robot PLACEs, the other robot's WAIT PATH should retreat to its side of the table if it is near the bin or holding an item.
- Do not let two robots PLACE in the same round.
- Do not PLACE into an occupied slot.
Prefer [Recommended Pack Plan] exactly unless environment feedback says it failed.
Output only the final EXECUTE block:
"""

PACK_CHAT_PROMPT="""Robots discuss to find the best strategy and path. When each robot talk, it first reflects on the task status and its own capability. 
Carefully consider [Environment Feedback]. Coordinate with others to plan and improve paths following the instructions. They talk in order [Alice],[Bob],[Alice],..., then, after they agreed, plan exactly one ACTION per robot, output an EXECUTE to summarize the plan and stop talking.
Their discussion and the final plan: """

class PackGroceryTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_pack.xml",
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

        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()  
        panda_config = PANDA_CONSTANTS.copy() 

        self.item_names = PACK_ITEM_NAMES

        super(PackGroceryTask, self).__init__(
            filepath=filepath,  
            task_objects=PACK_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=robotiq_config,
                panda=panda_config, 
            ),
            **kwargs
        ) 
        
        self.bin_slot_xposes = dict()
        for sname in PACK_BIN_SITE_NAMES:
            self.bin_slot_xposes[sname] = self.physics.data.site(sname).xpos.copy()

        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )
         
        self.align_threshold = 0.06
    
    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]: 
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]

        if target_name in self.item_names:
            sname = f"{target_name}_top"  
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xpos.copy() 
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass

        return ret

    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if target_name in self.item_names:
            sname = f"{target_name}_top" 
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xmat.copy().reshape(3, 3)
            ret = mat_to_quat(ret)
            if any([name in sname for name in ['apple', 'soda_can', 'milk']]):
                # change quat
                if agent_name == "Bob":
                    ret = np.array([1, 0, 0, 1])
                else:
                    ret = np.array([1, 0, 0, 0])
            if 'bin_' in target_name and agent_name == "Bob":
                ret = np.array([1, 0, 0, 1])
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass
        return ret 
    
    @property 
    def use_prepick(self):
        return False  

    @property
    def use_preplace(self):
        return False
    
    @property
    def waypoint_std_threshold(self):
        return 0.19

    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        
        bin_id = self.physics.model.body("bin").id
        bin_bottom_id = self.physics.model.body("bin_inside").id
        table_id = self.physics.model.body("table").id

        ret = [(table_id, bin_bottom_id)]
        all_body_ids = []
        for obj_name in self.item_names:
            body_ids = self.get_all_body_ids(obj_name)
            for body_id in body_ids:
                ret.append((body_id, bin_bottom_id))
                # ret.append((body_id, bin_id)) this makes direct path less likely
                ret.append((body_id, table_id))
                all_body_ids.append(body_id)

        ee_link_ids = self.robots["Alice"].ee_link_body_ids + self.robots["Bob"].ee_link_body_ids
        ee_link_ids = [_id for _id in ee_link_ids if _id != "panda_hand"]

        return ret 

    def get_graspable_objects(self):
        graspables = self.item_names.copy()
        return dict(
            Alice=graspables,
            Bob=graspables, 
        )

    def get_grasp_site(self, obj_name: str = "apple") -> Optional[str]:
        if obj_name in self.item_names:
            return f"{obj_name}_top"
        else:
            return None

    def get_object_joint_name(self, obj_name: str) -> str:
        return f"{obj_name}_joint"

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name] 

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.3, 1.6), y=(-0.4, 1.5), z=(0, 1))
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-1.3, 1.6), y=(0, 1.5), z=(0, 1))
        else:
            raise NotImplementedError
    
    def sample_initial_scene(self): 
        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'grid' in geom.name:
                low = geom.pos - geom.size
                high = geom.pos + geom.size
                tosample_panels.append(
                    (low, high)
                )
        assert len(tosample_panels) >= len(self.item_names), "Not enough grid positions to sample from"
        panel_idxs = self.random_state.choice(
            len(tosample_panels), 
            len(self.item_names),
            replace=False
            )
        for _idx, item_name in zip(panel_idxs, self.item_names):
            low, high = tosample_panels[_idx]
            new_pos = self.random_state.uniform(low, high) 
            new_pos[2] = self.physics.data.body(item_name).xpos[2] # height stays same!
            new_quat = Quaternion(
                axis=[0,0,1], 
                angle=self.random_state.uniform(low=0, high=2*np.pi)
                ) 
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            self.reset_body_pose(
                body_name=item_name,
                pos=new_pos,
                quat=new_quat,
            )  
            self.reset_qpos(
                jnt_name=f"{item_name}_joint",
                pos=new_pos,
                quat=new_quat,
            )
          
        self.physics.forward()
        self.physics.step(50)
    
    def get_obs(self) -> EnvState:
        contacts = self.get_contact()
        allow_objs = self.item_names + ["bin", "table"]
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in allow_objs]
        contacts["panda"] = [c for c in contacts["panda"] if c in allow_objs]

        obj_states = self.get_object_states(contact_dict=contacts)
        agent_states = dict()
        for agent_name, agent_constants in self.agent_configs.items():
            agent_state = self.get_agent_state(
                agent_constants, contact_dict=contacts
            ) 
            agent_states[agent_name] = agent_state
        kwargs = dict(
            objects=obj_states,
        )
        kwargs.update(agent_states)
        if self.render_point_cloud:
            point_cloud = self.get_point_cloud()
            kwargs['scene'] = point_cloud # NOTE: should include bboxes! 
        obs = EnvState(**kwargs)
         
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation" 
        return obs
    
    def get_reward_done(self, obs): 
        all_packed = True
        reward = 1
        for food in self.item_names:
            bin_coord = self.physics.data.body("bin").xpos[:2]
            dist = np.linalg.norm(obs.objects[food].xpos[:2] - bin_coord)
            if 'bin_inside' not in obs.objects[food].contacts and dist > self.align_threshold:
                all_packed = False 
                reward = 0
                break 
        return reward, all_packed

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix! 
        robotiq_link_names = self.agent_configs["ur5e_robotiq"]['all_link_names'] + ['ur5e_robotiq']
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c not in robotiq_link_names] 

        panda_link_names = self.agent_configs["panda"]['all_link_names'] + ["panda_right_finger", "panda_left_finger", "panda"]
        contacts["panda"] = [c for c in contacts['panda'] if c not in panda_link_names] 
        contacts["panda"].append("broom")

        return contacts

    def _agent_holding_item(self, obs: EnvState, agent_name: str) -> Optional[str]:
        robot_name = self.robot_name_map_inv[agent_name]
        contacts = getattr(obs, robot_name).contacts
        held_items = [c for c in contacts if c in self.item_names]
        return held_items[0] if len(held_items) > 0 else None

    def _agent_ee_pos(self, obs: EnvState, agent_name: str) -> np.ndarray:
        robot_name = self.robot_name_map_inv[agent_name]
        return getattr(obs, robot_name).ee_xpos.copy()

    def _is_item_packed(self, obs: EnvState, item_name: str) -> bool:
        bin_coord = self.physics.data.body("bin").xpos[:2]
        dist = np.linalg.norm(obs.objects[item_name].xpos[:2] - bin_coord)
        return 'bin_inside' in obs.objects[item_name].contacts or dist <= self.align_threshold

    def _occupied_slots(self, obs: EnvState) -> Set[str]:
        occupied = set()
        for item_name in self.item_names:
            if not self._is_item_packed(obs, item_name):
                continue
            item_xy = obs.objects[item_name].xpos[:2]
            nearest_slot, nearest_dist = min(
                (
                    (slot_name, np.linalg.norm(item_xy - slot_xpos[:2]))
                    for slot_name, slot_xpos in self.bin_slot_xposes.items()
                ),
                key=lambda x: x[1],
            )
            if nearest_dist < 0.18:
                occupied.add(nearest_slot)
        return occupied

    def _empty_slots(self, obs: EnvState) -> List[str]:
        occupied = self._occupied_slots(obs)
        return [slot for slot in PACK_BIN_SITE_NAMES if slot not in occupied]

    def _format_path(self, pts: List[np.ndarray]) -> str:
        return "[" + ", ".join(
            f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in pts
        ) + "]"

    def _wait_action(self, obs: EnvState, agent_name: str) -> str:
        pos = self._agent_ee_pos(obs, agent_name)
        return f"WAIT PATH {self._format_path([pos, pos, pos, pos])}"

    def _retreat_wait_action(self, obs: EnvState, agent_name: str) -> str:
        """WAIT while moving to a side parking pose away from the bin corridor.

        In action_and_path mode the parser treats Pack WAIT's last path point
        as the final waiting pose.  These side poses keep the inactive robot and
        any in-hand object out of the active robot's PLACE corridor.
        """
        start = self._agent_ee_pos(obs, agent_name)
        if agent_name == "Alice":
            goal = np.array([0.22, 0.06, 0.65])
        else:
            goal = np.array([0.35, 1.05, 0.65])
        safe_z = self._safe_z(start[2], goal[2])
        pts = [
            np.array([start[0], start[1], safe_z]),
            np.array([(2 * start[0] + goal[0]) / 3, (2 * start[1] + goal[1]) / 3, safe_z]),
            np.array([(start[0] + 2 * goal[0]) / 3, (start[1] + 2 * goal[1]) / 3, safe_z]),
            np.array([goal[0], goal[1], safe_z]),
        ]
        return f"WAIT PATH {self._format_path(pts)}"

    def _safe_z(self, *zs: float) -> float:
        return float(np.clip(max([0.65] + list(zs)), 0.62, 0.78))

    def _pick_action(self, obs: EnvState, agent_name: str, item_name: str) -> str:
        start = self._agent_ee_pos(obs, agent_name)
        target = self.get_target_pos(agent_name, item_name).copy()
        safe_z = self._safe_z(start[2], target[2] + 0.20)
        pts = [
            np.array([start[0], start[1], safe_z]),
            np.array([(2 * start[0] + target[0]) / 3, (2 * start[1] + target[1]) / 3, safe_z]),
            np.array([(start[0] + 2 * target[0]) / 3, (start[1] + 2 * target[1]) / 3, safe_z]),
            np.array([target[0], target[1], safe_z]),
        ]
        return f"PICK {item_name} PATH {self._format_path(pts)}"

    def _place_action(self, obs: EnvState, agent_name: str, item_name: str, slot_name: str) -> str:
        start = self._agent_ee_pos(obs, agent_name)
        target = self.get_target_pos(agent_name, slot_name).copy()
        # Keep tall objects well above the bin rim during the approach.  The
        # parser already raises the final place target for milk/cereal; these
        # waypoints protect the approach path.
        extra = 0.10 if item_name in ["milk", "cereal"] else 0.0
        safe_z = self._safe_z(start[2], target[2] + 0.28 + extra)
        pts = [
            np.array([start[0], start[1], safe_z]),
            np.array([(2 * start[0] + target[0]) / 3, (2 * start[1] + target[1]) / 3, safe_z]),
            np.array([(start[0] + 2 * target[0]) / 3, (start[1] + 2 * target[1]) / 3, safe_z]),
            np.array([target[0], target[1], safe_z]),
        ]
        return f"PLACE {item_name} {slot_name} PATH {self._format_path(pts)}"

    def _choose_items_for_empty_agents(self, obs: EnvState, empty_agents: List[str], reserved_items: Set[str]) -> Dict[str, str]:
        unpacked = [
            item for item in self.item_names
            if not self._is_item_packed(obs, item) and item not in reserved_items
        ]
        chosen = {}
        used = set()
        for agent_name in empty_agents:
            if len(unpacked) == 0:
                break
            ee = self._agent_ee_pos(obs, agent_name)
            candidates = [item for item in unpacked if item not in used]
            if len(candidates) == 0:
                break
            item = min(
                candidates,
                key=lambda name: np.linalg.norm(self.get_target_pos(agent_name, name)[:2] - ee[:2])
            )
            chosen[agent_name] = item
            used.add(item)
        return chosen

    def _choose_slot_for_agent(self, obs: EnvState, agent_name: str) -> Optional[str]:
        empty_slots = self._empty_slots(obs)
        if len(empty_slots) == 0:
            return None
        if agent_name == "Alice":
            preference = [
                "bin_front_left", "bin_front_middle", "bin_back_left",
                "bin_back_middle", "bin_front_right", "bin_back_right",
            ]
        else:
            preference = [
                "bin_back_right", "bin_back_middle", "bin_front_right",
                "bin_front_middle", "bin_back_left", "bin_front_left",
            ]
        for slot in preference:
            if slot in empty_slots:
                return slot
        return empty_slots[0]

    def _packed_count(self, obs: EnvState) -> int:
        return sum(1 for item in self.item_names if self._is_item_packed(obs, item))

    def _choose_place_agent(self, holding: Dict[str, Optional[str]]) -> Optional[str]:
        """Choose the held object that should be cleared from the workspace first.

        A waiting robot that is holding an object is still an obstacle for the
        active robot/RRT planner.  In practice Bob's held milk/cereal near the
        back/center blocks Alice's front-to-bin path, so when both are holding
        we bias toward placing Bob's item first.
        """
        holders = [agent for agent, item in holding.items() if item is not None]
        if len(holders) == 0:
            return None
        if len(holders) == 1:
            return holders[0]

        # Milk/cereal are tall and often become the blocking collision geometry.
        priority = {
            "milk": 4,
            "cereal": 4,
            "soda_can": 3,
            "bread": 2,
            "apple": 1,
            "banana": 1,
        }
        return max(
            holders,
            key=lambda agent: (
                priority.get(holding[agent], 0),
                1 if agent == "Bob" else 0,
            ),
        )

    def get_recommended_plan(self, obs: EnvState) -> Dict[str, str]:
        """Conservative pack policy.

        The original "parallel PICK then serial PLACE" rule is not sufficient:
        the serial WAIT robot may still be holding a bulky object, and that held
        object remains active collision geometry.  Therefore we bootstrap with a
        single first PICK, then use parallel PICKs only after one item has been
        packed, and when both robots are holding we place the more obstructive
        held item first.
        """
        holding = {
            agent: self._agent_holding_item(obs, agent)
            for agent in ["Alice", "Bob"]
        }
        plan = {
            "Alice": self._wait_action(obs, "Alice"),
            "Bob": self._wait_action(obs, "Bob"),
        }

        holders = [agent for agent, item in holding.items() if item is not None]
        empty_agents = [agent for agent, item in holding.items() if item is None]

        if len(holders) > 0:
            # At most one PLACE per round to avoid both arms entering the bin.
            place_agent = self._choose_place_agent(holding)
            slot = self._choose_slot_for_agent(obs, place_agent)
            if slot is not None:
                plan[place_agent] = self._place_action(obs, place_agent, holding[place_agent], slot)
            for agent in ["Alice", "Bob"]:
                if agent != place_agent:
                    plan[agent] = self._retreat_wait_action(obs, agent)
            return plan

        # On the first round, do not pick both central/tall items (usually
        # cereal and milk) at once.  That state caused Bob to WAIT while holding
        # milk in the middle of Alice's cereal PLACE corridor.  A single first
        # PICK costs one step but still allows a 10-step schedule:
        # 1 item alone + 2 paired items + 2 paired items + 1 item alone.
        if self._packed_count(obs) == 0 and len(empty_agents) == 2:
            first_agent = "Alice"
            first_item = self._choose_items_for_empty_agents(obs, [first_agent], set()).get(first_agent)
            if first_item is not None:
                plan[first_agent] = self._pick_action(obs, first_agent, first_item)
            return plan

        chosen = self._choose_items_for_empty_agents(obs, empty_agents, set())
        for agent, item in chosen.items():
            plan[agent] = self._pick_action(obs, agent, item)
        return plan

    def format_legal_actions_prompt(self, obs: EnvState) -> str:
        plan = self.get_recommended_plan(obs)
        packed = [item for item in self.item_names if self._is_item_packed(obs, item)]
        unpacked = [item for item in self.item_names if not self._is_item_packed(obs, item)]
        occupied = sorted(self._occupied_slots(obs))
        empty = self._empty_slots(obs)
        lines = [
            "[Pack State]",
            f"- Packed items: {packed if packed else ['none']}",
            f"- Unpacked items: {unpacked if unpacked else ['none']}",
            f"- Occupied bin slots: {occupied if occupied else ['none']}",
            f"- Empty bin slots: {empty if empty else ['none']}",
            f"- Alice holding: {self._agent_holding_item(obs, 'Alice') or 'nothing'}",
            f"- Bob holding: {self._agent_holding_item(obs, 'Bob') or 'nothing'}",
            "[Recommended Pack Plan]",
            "Follow this conservative plan exactly unless environment feedback says it failed:",
            "EXECUTE",
            f"NAME Alice ACTION {plan['Alice']}",
            f"NAME Bob ACTION {plan['Bob']}",
        ]
        return "\n".join(lines) + "\n"

    def central_plan_prompt(self, chat_history: List[str] = []):
        return PACK_PLAN_PROMPT 

    def get_action_prompt(self) -> str:
        return PACK_ACTION_SPACE

    def describe_object(self, obs, name):
        x,y,z = self.physics.data.site(f"{name}_top").xpos
        z += 0.05 # further avoid collision
        contacts = obs.objects[name].contacts 
        object_desp = f"{name}: ({x:.2f}, {y:.2f}, {z:.2f}), "
        if 'bin_inside' in contacts:
            dist_to_slot = [
                (
                    slot_name, np.linalg.norm(np.array([x,y]) - slot_xpos[:2])
                ) for slot_name, slot_xpos in self.bin_slot_xposes.items()

            ]
            slot_name = min(dist_to_slot, key=lambda x: x[1])[0]
            object_desp += f"inside slot {slot_name}"
        else:
            object_desp += f"on table"
        return object_desp

    def describe_robot_state(self, obs, robot_name):
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        contacts = robot_state.contacts 
        contacts = [c for c in contacts if c in self.item_names]
        obj = contacts[0] if len(contacts) > 0 else "nothing"
        agent_name = self.robot_name_map[robot_name]
        robot_desp = f"{agent_name}'s gripper: ({x:.2f}, {y:.2f}, {z:.2f}), holding {obj}" 
        return robot_desp
    
    def describe_obs(self, obs: EnvState):
        full_desp =  "[Scene description]\n" 
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        full_desp += f"robots must move lower than 0.8 but higher than table height {table_height:.2f}\n"
        for name in self.item_names:
            full_desp += self.describe_object(obs, name) + "\n"

        for slot_name, slot_xpos in self.bin_slot_xposes.items():
            x, y, z = slot_xpos
            full_desp += f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
 
        for robot_name, agent_name in self.robot_name_map.items():
            full_desp += self.describe_robot_state(obs, robot_name) + "\n"
            
        return full_desp 
    
    def describe_task_context(self):
        return PACK_TASK_CONTEXT
    
    def get_agent_prompt(self, obs, agent_name):        
        robot_name = self.get_robot_name(agent_name)
        other_robot = "Alice" if agent_name == "Bob" else "Bob"
        object_desp = "\n".join([self.describe_object(obs, name) for name in self.item_names])

        table_height = self.physics.data.body("table_top").xpos[2] + 0.15 
        robot_desp = self.describe_robot_state(obs, robot_name).replace(f"{agent_name}'s", "Your")
        slot_desp = "\n".join(
            [
                f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})" for slot_name, (x,y,z) in self.bin_slot_xposes.items()
            ]
            )

        agent_prompt = f"""
You are {agent_name}, you and robot {other_robot} each stands at a different side of the table, and together you must put all the grocery items into a bin.
Locations of slots in the bin:
{slot_desp}
At current round:
You see the following objects:
{object_desp}
{robot_desp}
Your gripper must move higher than these objects and higher than table height {table_height:.2f}, but move lower than 0.8.
Never forget you are {agent_name}!
Think step-by-step about the task and {other_robot}'s response. Carefully check and correct {other_robot} if they made a mistake. 
Discuss with {other_robot} to come up with the best plan and smooth, collision-free paths. 
Improve your paths if given [Environment Feedback], choose a different object or target slot if needed.

When you respond, tell {other_robot} about your status. Respond very concisely but informatively, and do not repeat what others have said.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction] and [Path Plan Instruction].
"""
        return agent_prompt
    
    def get_task_feedback(self, llm_plan, pose_dict): 
        feedback = ""
        obs = self.get_obs()
        actions = llm_plan.action_strs
        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' not in action_str and 'PLACE' not in action_str and 'WAIT' not in action_str:
                feedback += f"{agent_name}'s ACTION is invalid, can only PICK, PLACE, or WAIT\n"
            if 'PLACE' in action_str:
                parts = action_str.split('PLACE', 1)[1].split('PATH', 1)[0].strip().split()
                if len(parts) < 2:
                    feedback += f"{agent_name}'s PLACE action must be PLACE <obj> <bin_slot> PATH <path>\n"
                    continue
                slot_name = parts[1]
                if slot_name == "bin" or slot_name not in PACK_BIN_SITE_NAMES:
                    feedback += f"{agent_name}'s PLACE target must be an explicit empty bin slot, not {slot_name}\n"
                elif slot_name in self._occupied_slots(obs):
                    feedback += f"{agent_name} cannot PLACE into occupied slot {slot_name}\n"

        num_place = sum('PLACE' in action for action in actions.values())
        num_pick = sum('PICK' in action for action in actions.values())
        if num_place > 1:
            feedback += "At most one robot may PLACE per round to avoid collision near the bin; the other robot should WAIT.\n"
        if num_place > 0 and num_pick > 0:
            feedback += "Do not mix PLACE and PICK in the same round for pack; PLACE robot uses the bin area and the other robot should WAIT.\n"

        holding = {
            agent: self._agent_holding_item(obs, agent)
            for agent in ["Alice", "Bob"]
        }
        holders = [agent for agent, item in holding.items() if item is not None]
        if len(holders) > 0:
            place_agents = [agent for agent, action in actions.items() if 'PLACE' in action]
            if len(place_agents) == 0:
                feedback += "A robot is already holding an item; one holding robot should PLACE it into an empty bin slot while the other robot WAITs.\n"
            for agent in holders:
                if 'PICK' in actions.get(agent, ''):
                    feedback += f"{agent} is already holding {holding[agent]} and cannot PICK another item.\n"
            for agent, held in holding.items():
                if held is None and 'WAIT' not in actions.get(agent, ''):
                    feedback += f"{agent} is empty-handed but another robot is placing; {agent} should WAIT to avoid collision.\n"
        else:
            if num_place > 0:
                feedback += "No robot is holding an item, so robots should PICK unpacked items, not PLACE.\n"
            picked = []
            for action in actions.values():
                if 'PICK' in action:
                    picked.append(action.split('PICK', 1)[1].split('PATH', 1)[0].strip().split()[0])
            duplicates = sorted({item for item in picked if picked.count(item) > 1})
            if duplicates:
                feedback += f"Robots cannot PICK the same item in one round: {duplicates}\n"
        if all('WAIT' in action for action in actions.values()) and not all(self._is_item_packed(obs, item) for item in self.item_names):
            feedback += "All robots cannot WAIT while unpacked items remain.\n"
        return feedback
 
 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image 
    env = PackGroceryTask()
    obs = env.reset()
    print(env.describe_obs(obs))
    print(env.get_agent_prompt(obs, "Alice"))
    print(env.get_agent_prompt(obs, "Bob"))
    breakpoint()
    print(obs.ur5e_robotiq.ee_xquat)
    img=env.physics.render(camera_id="teaser", height=480, width=600)
    im = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
    breakpoint()
