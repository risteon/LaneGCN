import numpy as np

import logging
import pathlib
import operator
import csv
import re
from collections import defaultdict


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class Agent:
    def __init__(self):
        # timestamp -> state
        self.states = {}
        self.stamps = {}
        self.time_range = None
        self.states_dense = None
        self.dims = None
        self.classification = defaultdict(lambda: float(0.0))
        self.agent_type = None
        self.pos_range = None

    def __len__(self):
        return len(self.states)

    def index_states(self, env_id_index_map: {int: int}):
        """Finalize agent, create dense arrays

        :param env_id_index_map: Map env ID to index in dense structure over time
        :return:
        """
        slots = np.asarray([env_id_index_map[x] for x in self.states.keys()])
        self.time_range = slots.min(), slots.max() + 1
        self.states_dense = np.full(
            shape=(self.time_range[1] - self.time_range[0], 10), fill_value=float("nan")
        )
        for k, v in self.states.items():
            self.states_dense[env_id_index_map[k] - self.time_range[0], :] = v

        self.dims = np.median(self.states_dense[:, 7:10], 0)
        self.states_dense = self.states_dense[:, :7]
        self.pos_range = np.stack(
            (
                self.states_dense[:, 0:2].min(axis=0),
                self.states_dense[:, 0:2].max(axis=0),
            ),
            axis=0,
        )

    def finalize_classification(self, agent_classification_mapping: {str: str}) -> bool:
        self.classification = sorted(
            list(self.classification.items()), key=operator.itemgetter(1), reverse=True
        )
        try:
            self.agent_type = agent_classification_mapping[self.classification[0][0]]
            return True
        except KeyError:
            self.agent_type = "<invalid>"
            return False

    def update_classification(self, classification: {str: float}):
        for k, v in classification.items():
            self.classification[k] += v


class Lane:
    def __init__(self, lane_id, center_from, center_to):
        """
        # set up some lanes {lane_id -> lane}
        # lane.centerline: ndarray [N, 2(xy)], world coords, N ~= 10 (?)
        #                  Euclidean distance = 1.58
        # lane.predecessors: (can be None) list[int] (lane_ids)
        # lane.successors: (can be None) list[int] (lane_ids)
        # lane.l_neighbor_id: Union(int, None) (lane_id)
        # lane.r_neighbor_id: Union(int, None) (lane_id)
        # lane.has_traffic_control: bool
        # lane.id: int
        # lane.is_intersection: bool
        # lane.turn_direction: str
        # {'RIGHT', 'LEFT', 'NONE' (is_intersection == False)}

        # seems to be unused
        # lane_polygon: [ndarray, 21, 3(xyz)] z ~= -21.7]
        """
        # meta
        self.id = lane_id
        self.is_intersection = False
        self.turn_direction = "NONE"
        self.has_traffic_control = False

        # topology
        self.l_neighbor_id = None
        self.r_neighbor_id = None
        self.predecessors = None
        self.successors = None

        # geometry
        # straight line, fixed distance between vertices
        distance = np.linalg.norm(center_to - center_from)
        delta = 1.5
        n = int((distance // delta)) + 2
        x = np.linspace(center_from[0], center_to[0], n)
        y = np.linspace(center_from[1], center_to[1], n)

        self.centerline = np.stack((x, y), axis=-1)


class Scene:
    class ConstantAcc:
        def __init__(self, s0, v0, a):
            self.s0 = s0
            self.v0 = v0
            self.a = a

        def __call__(self, t):
            return self.s0 + self.v0 * t + 0.5 * self.a * t ** 2

    def __init__(self):

        self.agent_classification_mapping = {
            "PassengerCar": "Vehicle",
            "LargeVehicle": "Vehicle",
            "VulnerableRoadUser": "Pedestrian",
            "Pedestrian": "Pedestrian",
            "RidableVehicle": "Pedestrian",
        }

        self.agents = defaultdict(lambda: Agent())
        self.env_model_stamps = {}
        self.agent_classifications = set()

        self.agent_counts_at_t = None
        self.env_id_index_map = None
        self.env_id_indices = None

        self.lanes = {}

    @staticmethod
    def parse_classification(labels: str, confidence: str):
        labels = labels[1:-1].split("|")
        confidence = [float(x) for x in confidence[1:-1].split("|")]
        if len(labels) != len(confidence):
            raise ValueError(f"label confidence mismatch: {labels}, {confidence}")
        return dict(zip(labels, confidence))

    @classmethod
    def create_simple(cls, speed=30.0 / 3.6, class_label="PassengerCar"):
        scene = cls()
        scene.name = "Single agent."

        # single agent, 10 sec straight movement, 10 Hz
        # don't actually need speed and acc (resampled anyway)
        for em_id, stamp in enumerate(np.arange(0.0, 5.0, 0.1)):

            agent_id = 0
            scene.agent_classifications.update({class_label: 0.99})
            x_pos = stamp * speed
            state_vec = [x_pos, 0.0, 0.0, 0.0, 0.0, 0.0]
            yaw = 0.0
            length = 4.5
            height = 1.5
            width = 1.8

            state_vec = np.asarray(state_vec + [yaw, length, width, height])

            scene.agents[agent_id].states[em_id] = state_vec
            scene.agents[agent_id].stamps[em_id] = stamp
            scene.agents[agent_id].update_classification({class_label: 0.99})
            scene.env_model_stamps[em_id] = stamp

        scene.index_timestamps()
        scene.process_agents()
        scene.finalize()

        # single lane for agent to drive on
        lane_delta = 10.0
        scene.create_lane(
            np.asarray([-lane_delta, 0.0]), np.asarray([5.0 * speed + lane_delta, 0.0])
        )
        return scene

    def create_lane(self, center_from, center_to):
        lane_id = len(self.lanes)
        self.lanes[lane_id] = Lane(lane_id, center_from, center_to)
        return lane_id

    @classmethod
    def create_follow(cls):
        scene = cls()
        scene.name = "Single agent."
        class_label = "PassengerCar"

        s_follow = cls.ConstantAcc(0.0, 30.0 / 3.6, -0.1)
        s_lead = cls.ConstantAcc(20.0, 15.0 / 3.6, 0.0)

        # two agents, 10 sec straight movement, 10 Hz
        # don't actually need speed and acc (resampled anyway)
        for em_id, stamp in enumerate(np.arange(0.0, 5.0, 0.1)):
            agent_id = 0
            scene.agent_classifications.update({class_label: 0.99})
            state_vec = [s_follow(stamp), 0.5, 0.0, 0.0, 0.0, 0.0]
            yaw = 0.0
            length = 4.5
            height = 1.5
            width = 1.8

            state_vec = np.asarray(state_vec + [yaw, length, width, height])

            scene.agents[agent_id].states[em_id] = state_vec
            scene.agents[agent_id].stamps[em_id] = stamp
            scene.agents[agent_id].update_classification({class_label: 0.99})
            scene.env_model_stamps[em_id] = stamp

            agent_id = 1
            scene.agent_classifications.update({class_label: 0.99})
            state_vec = [s_lead(stamp), -0.5, 0.0, 0.0, 0.0, 0.0]
            yaw = 0.0
            length = 4.5
            height = 1.5
            width = 1.8

            state_vec = np.asarray(state_vec + [yaw, length, width, height])

            scene.agents[agent_id].states[em_id] = state_vec
            scene.agents[agent_id].stamps[em_id] = stamp
            scene.agents[agent_id].update_classification({class_label: 0.99})
            scene.env_model_stamps[em_id] = stamp

        scene.index_timestamps()
        scene.process_agents()
        scene.finalize()

        # single lane for agent to drive on
        lane_delta = 10.0
        scene.create_lane(
            np.asarray([-lane_delta, 0.0]), np.asarray([s_lead(5.0) + lane_delta, 0.0])
        )

        return scene

    @classmethod
    def create_pedestrian_interaction(cls):
        scene = cls()
        scene.name = "Single agent."
        class_label_a = "PassengerCar"
        class_label_b = "Pedestrian"

        s_follow = cls.ConstantAcc(0.0, 20.0 / 3.6, -0.15)
        s_ped = cls.ConstantAcc(-3.5, 3.5 / 3.6, 0.0)

        # two agents, 10 sec straight movement, 10 Hz
        # don't actually need speed and acc (resampled anyway)
        for em_id, stamp in enumerate(np.arange(0.0, 5.0, 0.1)):
            # agent_id = 0
            # scene.agent_classifications.update({class_label_a: 0.99})
            # state_vec = [s_follow(stamp), 0.5, 0.0, 0.0, 0.0, 0.0]
            # yaw = 0.0
            # length = 4.5
            # height = 1.5
            # width = 1.8
            #
            # state_vec = np.asarray(state_vec + [yaw, length, width, height])
            #
            # scene.agents[agent_id].states[em_id] = state_vec
            # scene.agents[agent_id].stamps[em_id] = stamp
            # scene.agents[agent_id].update_classification({class_label_a: 0.99})
            # scene.env_model_stamps[em_id] = stamp

            agent_id = 1
            scene.agent_classifications.update({class_label_b: 0.99})
            state_vec = [18.0, s_ped(stamp), 0.0, 0.0, 0.0, 0.0]
            yaw = -np.pi / 2.0
            length = 0.7
            height = 1.75
            width = 0.5

            state_vec = np.asarray(state_vec + [yaw, length, width, height])

            scene.agents[agent_id].states[em_id] = state_vec
            scene.agents[agent_id].stamps[em_id] = stamp
            scene.agents[agent_id].update_classification({class_label_b: 0.99})
            scene.env_model_stamps[em_id] = stamp

        scene.index_timestamps()
        scene.process_agents()
        scene.finalize()
        return scene

    @classmethod
    def parse_from_csv(cls, agent_csv: pathlib.Path):

        scene = cls()
        scene.name = agent_csv.stem

        # todo can't figure out how to have '\{([^\{\}]+)\}' 6 times?
        p = re.compile(
            r"\[\{([^\{\}]+)\}\{([^\{\}]+)\}\{([^\{\}]+)\}\{([^\{\}]+)\}\{([^\{\}]+)\}\{([^\{\}]+)\}\]"
        )

        with open(agent_csv) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            for row in reader:
                state_vec = row["state_vec [-]"]
                em_id = int(row["em_id [-]"])
                stamp = float(row["em_time [s]"])
                # what about "dyn_object_id [-]"?
                agent_id = int(row["tado_id [-]"])
                classification_label = row["labels_bayesian [-]"]
                classification_confidence = row["labels_bayesian_confidence [-]"]
                classification = scene.parse_classification(
                    classification_label, classification_confidence
                )
                scene.agent_classifications.update(classification.keys())

                m = p.fullmatch(state_vec)
                if m is None:
                    raise ValueError(state_vec)
                state_vec = [float(m[i + 1]) for i in range(6)]

                yaw = float(row["yaw_abs [rad]"])
                length = float(row["obb_length [m]"])
                height = float(row["obb_height [m]"])
                width = float(row["obb_width [m]"])

                state_vec = np.asarray(state_vec + [yaw, length, width, height])

                scene.agents[agent_id].states[em_id] = state_vec
                scene.agents[agent_id].stamps[em_id] = stamp
                scene.agents[agent_id].update_classification(classification)
                scene.env_model_stamps[em_id] = stamp

                # if np.any(state_vec[2:] != 0.0):
                #     print(state_vec)

        scene.index_timestamps()
        scene.process_agents()
        scene.finalize()
        return scene

    def to_lanegcn(self, agent_index=0, t_pred=20):

        """
        AGENT: max interesting score in argoverse. Make prediction
        OTHERS: all other agents
        AV: autonomous vehicle
        """

        import pandas as pd

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = pd.DataFrame(
            columns=[
                "TIMESTAMP",
                "TRACK_ID",
                "OBJECT_TYPE",
                "X",
                "Y",
                "CITY_NAME",
            ]
        )

        # node_type_map = {
        #     "Vehicle": "AGENT",
        #     "Pedestrian": "OTHERS",
        # }

        city = "dummy"

        # trajs should always have 20 steps (= 2s at 10 Hz) history
        # use t_pred to keep (t_pred - 20 : t_pred) steps for every agent
        n_states_past = 20
        n_states_future = 30
        for agent_id, agent in self.agents.items():

            agent_begin = t_pred - agent.time_range[0] - n_states_past
            if agent_begin < 0:
                raise ValueError("Not enough states for agent.")
            if agent_begin + n_states_past >= agent.time_range[1]:
                raise ValueError("Not enough states for agent.")

            for i, state in enumerate(
                agent.states_dense[
                    agent_begin : agent_begin + n_states_past + n_states_future
                ]
            ):

                data_point = pd.Series(
                    {
                        "TIMESTAMP": agent.stamps[i],  # float64 in secs
                        "TRACK_ID": f"{agent_id:03d}",
                        "OBJECT_TYPE": "AGENT" if agent_id == agent_index else "OTHERS",
                        "X": state[0],
                        "Y": state[1],
                        "CITY_NAME": city,
                    }
                )
                df = df.append(data_point, ignore_index=True)

        agt_ts = np.sort(np.unique(df["TIMESTAMP"].values))
        mapping = {ts: i for i, ts in enumerate(agt_ts)}

        trajs = np.concatenate(
            (df.X.to_numpy().reshape(-1, 1), df.Y.to_numpy().reshape(-1, 1)), 1
        )

        steps = [mapping[x] for x in df["TIMESTAMP"].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(["TRACK_ID", "OBJECT_TYPE"]).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index("AGENT")
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data["city"] = city
        data["trajs"] = [agt_traj] + ctx_trajs
        data["steps"] = [agt_step] + ctx_steps

        data["scene"] = self
        return data

    def index_timestamps(self):
        xx = sorted(list(self.env_model_stamps.items()), key=operator.itemgetter(1))
        # check deltas in time stamps
        stamps = np.asarray([x[1] for x in xx])
        deltas = np.ediff1d(stamps)

        # Todo how to handle large/small diffs?
        logger.info(f"Largest delta in time stamps is {np.max(deltas)}")
        logger.info(f"Smallest delta in time stamps is {np.min(deltas)}")

        self.env_id_index_map = {x[1][0]: x[0] for x in enumerate(xx)}
        self.env_id_indices = np.asarray([x[0] for x in xx])

    def process_agents(self):
        # todo configuration
        # self.agents = {k: v for k, v in self.agents.items() if len(v) >= 10}
        self.agents = {k: v for k, v in self.agents.items() if len(v) >= 3}
        self.agents = {
            k: (v.finalize_classification(self.agent_classification_mapping), v)
            for k, v in self.agents.items()
        }
        self.agents = {k: v[1] for k, v in self.agents.items() if v[0]}

        for agent in self.agents.values():
            agent.index_states(self.env_id_index_map)

    def finalize(self):
        # Agent count data
        # xx = np.asarray(
        #     [(self.env_id_index_map[a], b) for a, b in self.agent_counts_at_t.items()]
        # )
        # agent_counts_dense = np.zeros(shape=(len(self.env_id_indices),), dtype=np.int32)
        # agent_counts_dense[xx[:, 0]] = xx[:, 1]
        self.agent_counts_at_t = np.zeros(
            shape=(
                len(
                    self.env_id_indices,
                )
            ),
            dtype=np.int32,
        )
        for agent in self.agents.values():
            self.agent_counts_at_t[agent.time_range[0] : agent.time_range[1]] += 1
