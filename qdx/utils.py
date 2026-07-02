from datetime import datetime
import itertools
from itertools import combinations
from pathlib import Path

import chex
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from more_itertools import distinct_permutations
import scipy.special as ss
import stim

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on the runtime env
    yaml = None

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GraphPadding
from qdx.make_train import make_actor_critic
from qdx.simulators import TableauSimulator
from qdx.simulators.clifford_gates import CliffordGates


BASE_CONFIG = {
    "MODEL": "GNN",
    "ENV_TYPE": "STANDARD",
    "D": 3,
    "MAX_STEPS": 50,
    "WHICH_GATES": ("cx", "h", "s", "sqrt_x", "cz", "sqrt_xx"),
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "VALIDATION_SOFTNESS": 3,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1.0e-3,
    "NUM_ENVS_PER_TASK": 16,
    "NUM_STEPS": 50,
    "TOTAL_TIMESTEPS": 2_000_000,
    "UPDATE_EPOCHS": 3,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "COMPUTE_METRICS": True,
    "GNN_HIDDEN_DIM": 64,
    "GNN_RELATION_DIM": 8,
    "GNN_GATE_DIM": 8,
    "GNN_NUM_LAYERS": 3,
}

DEFAULT_TRAIN_TASKS = ((5, 1, 3),)
DEFAULT_VALIDATION_TASKS = ((5, 1, 3), (6, 1, 3))
DEFAULT_GRAPHS = ("All-to-All", "NN-1", "NN-2")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"


class Utils:

    def __init__(self, n, k, gates, softness):
        self.n_qubits_physical = n
        self.n_qubits_logical = k

        tableau_simulator = TableauSimulator(self.n_qubits_physical)
        for gate in gates:
            eval(f"tableau_simulator{gate}")

        self.tableau = tableau_simulator.current_tableau[0]
        self.generate_S_structure(softness)
        self.S = self.generate_S(
            self.tableau[self.n_qubits_physical + self.n_qubits_logical :]
        )
        self.Omega = jnp.kron(
            jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8),
            jnp.eye(self.n_qubits_physical, dtype=jnp.uint8),
        )

    def error_operators(self, eval_weight) -> chex.Array:
        error_list = [1, 2, 3]
        prod = list(
            list(tup)
            for tup in itertools.combinations_with_replacement(error_list, eval_weight)
        )
        prod = [pr + [0] * (self.n_qubits_physical - eval_weight) for pr in prod]
        result = list(
            list(distinct_permutations(pr, self.n_qubits_physical)) for pr in prod
        )
        error_structure = list(itertools.chain(*result))
        return jnp.array(
            [
                jnp.array(stim.PauliString(pauli).to_numpy()).flatten()
                for pauli in error_structure
            ],
            dtype=jnp.uint8,
        )

    def generate_S_structure(self, softness):
        num = self.n_qubits_physical - self.n_qubits_logical
        soft_elements = int(sum(ss.binom(num, i) for i in range(1, softness + 1)))
        s_struct = np.zeros((soft_elements, num), dtype=int)

        start_idx = 0
        for weight in range(1, softness + 1):
            comb = list(combinations(range(num), weight))
            indices = np.array(comb)
            for offset, idx in enumerate(indices):
                s_struct[start_idx + offset, idx] = 1
            start_idx += offset + 1

        assert np.prod(np.any(s_struct, axis=1)), "There is a row with all zeroes"
        self.S_struct = jnp.array(s_struct, dtype=jnp.uint8)

    def generate_S(self, tableau):
        return (self.S_struct @ tableau) % 2

    def remove_rows(self, arr, row_indices):
        mask = jnp.ones(arr.shape, dtype=int)
        mask = mask.at[jnp.array(row_indices)].set(0)
        mask = mask.astype(bool)

        row_count = jnp.array(row_indices).shape[0]
        total_rows, width = arr.shape
        return arr[mask].reshape(total_rows - row_count, width)

    def check_KL(self, E_mu):
        inS = jax.vmap(jnp.logical_xor, in_axes=(None, 0))(self.S, E_mu)
        inS = jnp.prod(jnp.logical_not(inS), axis=-1)

        num_KL = (
            len(E_mu)
            - jnp.sum(
                jnp.any(
                    (
                        (E_mu @ self.Omega)
                        @ self.tableau[
                            self.n_qubits_physical + self.n_qubits_logical :
                        ].T
                    )
                    % 2,
                    axis=1,
                ),
                axis=0,
            )
            - jnp.sum(inS)
        )
        return num_KL

    def check_KL_cZ(self, E_mu, cZ):
        h_x = E_mu[:, : self.n_qubits_physical]
        h_z = E_mu[:, self.n_qubits_physical :]
        h_y = h_x * h_z
        effective_weights = (
            jnp.sum(h_x - h_y, axis=1)
            + jnp.sum(h_y, axis=1)
            + cZ * jnp.sum(h_z - h_y, axis=1)
        )

        tableau = self.tableau[self.n_qubits_physical + self.n_qubits_logical :]
        inS = jax.vmap(jnp.logical_xor, in_axes=(None, 0))(self.S, E_mu)
        inS = jnp.prod(jnp.logical_not(inS), axis=-1)

        kl1 = effective_weights * jnp.logical_not(
            jnp.any((E_mu @ self.Omega @ tableau.T) % 2, axis=1)
        )
        kl2 = effective_weights * jnp.sum(inS, axis=1)
        kls = kl1 - kl2

        non_zero_ids = jnp.nonzero(kls)[0]
        if len(kls[non_zero_ids]) > 0:
            smallest_undetected_effective_weight = jnp.min(kls[non_zero_ids])
        else:
            smallest_undetected_effective_weight = 100

        return round(float(smallest_undetected_effective_weight), 2)


def all_to_all_graph(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def nn_1_graph(n):
    graph = []
    for i in range(n - 1):
        graph.append((i, i + 1))
        graph.append((i + 1, i))
    return graph


def nn_2_graph(n):
    graph = nn_1_graph(n)
    for i in range(n - 2):
        graph.append((i, i + 2))
        graph.append((i + 2, i))
    return graph


def normalize_graph_name(graph_name):
    graph_name = str(graph_name).strip()
    if graph_name not in DEFAULT_GRAPHS:
        supported = ", ".join(DEFAULT_GRAPHS)
        raise ValueError(f"unsupported graph {graph_name!r}; choose from {supported}")
    return graph_name


def build_hardware_graph(n, graph_name):
    graph_name = normalize_graph_name(graph_name)
    if graph_name == "All-to-All":
        return all_to_all_graph(n)
    if graph_name == "NN-1":
        return nn_1_graph(n)
    if graph_name == "NN-2":
        return nn_2_graph(n)
    raise ValueError(f"unsupported graph {graph_name!r}")


def make_env(n, k, d, config, graph_padding, graph_name=None):
    graph_name = normalize_graph_name(
        config["GRAPH"] if graph_name is None else graph_name
    )
    gates = CliffordGates(n)
    gate_set = [getattr(gates, gate_name) for gate_name in config["WHICH_GATES"]]
    return GraphCodeDiscovery(
        n,
        k,
        d,
        gate_set,
        graph=build_hardware_graph(n, graph_name),
        max_steps=config["MAX_STEPS"],
        lbda=config["LAMBDA"],
        pI=config["P_I"],
        softness=config["SOFTNESS"],
        graph_padding=graph_padding,
    )


def build_task_config(base_config, task):
    task_config = dict(base_config)
    task_config["D"] = task["d"]
    task_config["GRAPH"] = task["graph"]
    return task_config


def make_task_env(task, config, graph_padding):
    return make_env(
        task["n"],
        task["k"],
        task["d"],
        config,
        graph_padding,
        graph_name=task["graph"],
    )


def normalize_task_specs(task_specs, default_task_specs, graphs):
    raw_specs = default_task_specs if task_specs is None else task_specs
    default_graphs = tuple(
        dict.fromkeys(normalize_graph_name(graph_name) for graph_name in graphs)
    )
    if not default_graphs:
        raise ValueError("at least one graph is required")

    normalized = []
    for task in raw_specs:
        task_graph = None
        task_graphs = None
        if isinstance(task, dict):
            n = task["n"]
            k = task["k"]
            d = task.get("d", task.get("target_distance", BASE_CONFIG["D"]))
            task_graph = task.get("graph")
            task_graphs = task.get("graphs")
        else:
            if len(task) not in (3, 4):
                raise ValueError(
                    "task specs must be 3-tuples (n, k, d) or 4-tuples (n, k, d, graph)"
                )
            n, k, d = task[:3]
            if len(task) == 4:
                task_graph = task[3]

        n = int(n)
        k = int(k)
        d = int(d)
        if n <= 0 or k <= 0 or d <= 0:
            raise ValueError("task values must be positive integers")
        if k >= n:
            raise ValueError(f"task {(n, k, d)} must satisfy k < n")

        if task_graphs is not None:
            graph_inputs = [task_graphs] if isinstance(task_graphs, str) else task_graphs
            task_graph_names = tuple(
                dict.fromkeys(
                    normalize_graph_name(graph_name) for graph_name in graph_inputs
                )
            )
            if not task_graph_names:
                raise ValueError("at least one graph is required")
        elif task_graph is not None:
            task_graph_names = (normalize_graph_name(task_graph),)
        else:
            task_graph_names = default_graphs

        for graph_name in task_graph_names:
            normalized.append({"n": n, "k": k, "d": d, "graph": graph_name})

    if not normalized:
        raise ValueError("at least one task is required")
    return normalized


def build_graph_padding(task_specs):
    if not task_specs:
        raise ValueError("at least one task is required to build graph padding")
    max_n = max(task["n"] for task in task_specs)
    max_stabilizers = max(task["n"] - task["k"] for task in task_specs)
    max_hardware_edges = max(
        len(build_hardware_graph(task["n"], task["graph"])) for task in task_specs
    )
    return GraphPadding(
        n_max=max_n,
        stabilizers_max=max_stabilizers,
        hardware_edges_max=max_hardware_edges,
    )


def graph_padding_to_dict(graph_padding):
    return {
        "n_max": int(graph_padding.n_max),
        "stabilizers_max": int(graph_padding.resolved_stabilizers_max),
        "hardware_edges_max": int(graph_padding.resolved_hardware_edges_max),
        "actions_max": (
            None if graph_padding.actions_max is None else int(graph_padding.actions_max)
        ),
    }


def format_task(task):
    return f"GRAPH={task['graph']} N={task['n']} K={task['k']} D={task['d']}"


def distance_error_stats_up_to_target(
    n, k, gates, target_distance, softness=None
):
    max_softness = n - k
    if max_softness < 1:
        raise ValueError("distance checks require n > k")
    resolved_softness = (
        max_softness
        if softness is None
        else max(1, min(int(softness), max_softness))
    )

    utilities = Utils(n, k, gates, softness=resolved_softness)
    distance_stats = []
    first_failure = target_distance + 1
    for weight in range(1, target_distance + 1):
        error_operators = utilities.error_operators(weight)
        error_count = int(utilities.check_KL(error_operators))
        total_count = int(error_operators.shape[0])
        error_rate = error_count / total_count if total_count else 0.0
        distance_stats.append(
            {
                "d": weight,
                "error_count": error_count,
                "total_count": total_count,
                "error_count_over_total": f"{error_count}/{total_count}",
                "error_rate": error_rate,
            }
        )
        if error_count != 0 and first_failure == target_distance + 1:
            first_failure = weight
    return first_failure, distance_stats


def format_distance_stats(distance_stats):
    if not distance_stats:
        return "distance_stats=[]"
    formatted = "; ".join(
        (
            f"d={item['d']} "
            f"error_count/total_count={item['error_count_over_total']} "
            f"error_rate={item['error_rate']:.2%}"
        )
        for item in distance_stats
    )
    return f"distance_stats=[{formatted}]"


def aggregate_distance_stats(results):
    stats_by_d = {}
    for result in results:
        for item in result.get("distance_stats", []) or []:
            d = item["d"]
            summary = stats_by_d.setdefault(
                d,
                {
                    "d": d,
                    "error_count": 0,
                    "total_count": 0,
                },
            )
            summary["error_count"] += item["error_count"]
            summary["total_count"] += item["total_count"]

    aggregated = []
    for d in sorted(stats_by_d):
        error_count = stats_by_d[d]["error_count"]
        total_count = stats_by_d[d]["total_count"]
        aggregated.append(
            {
                "d": d,
                "error_count": error_count,
                "total_count": total_count,
                "error_count_over_total": f"{error_count}/{total_count}",
                "error_rate": error_count / total_count if total_count else 0.0,
            }
        )
    return aggregated


def default_output_dir(run_name="demo_multitask_nkdg"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(__file__).resolve().parents[1] / "outputs" / f"{run_name}_{timestamp}"


def load_yaml_config(config_path):
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML configs. "
            "Install it in the active environment, for example with "
            "`conda run -n qdx pip install PyYAML`."
        )

    config_path = Path(config_path).expanduser()
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_run_settings(config_path):
    raw_config = load_yaml_config(config_path)
    config = {
        key: raw_config.get(key.lower(), raw_config.get(key, value))
        for key, value in BASE_CONFIG.items()
    }
    if "WHICH_GATES" in config and not isinstance(config["WHICH_GATES"], tuple):
        config["WHICH_GATES"] = tuple(config["WHICH_GATES"])

    graph_names = raw_config.get("graphs", DEFAULT_GRAPHS)
    graph_names = [graph_names] if isinstance(graph_names, str) else graph_names
    graphs = tuple(
        dict.fromkeys(normalize_graph_name(graph_name) for graph_name in graph_names)
    )
    train_tasks = normalize_task_specs(
        raw_config.get("train_tasks"), DEFAULT_TRAIN_TASKS, graphs
    )
    validation_tasks = normalize_task_specs(
        raw_config.get("validation_tasks"), DEFAULT_VALIDATION_TASKS, graphs
    )
    output_dir_value = raw_config.get("output_dir")
    output_dir = (
        default_output_dir()
        if output_dir_value in (None, "")
        else Path(output_dir_value).expanduser()
    )

    return {
        "config": config,
        "config_path": str(Path(config_path).expanduser()),
        "graphs": graphs,
        "train_tasks": train_tasks,
        "validation_tasks": validation_tasks,
        "skip_validation": raw_config.get("skip_validation", False),
        "skip_distance": raw_config.get("skip_distance", False),
        "dry_run": raw_config.get("dry_run", False),
        "output_dir": output_dir,
    }


def load_params_from_path(params_path, config, reference_task, graph_padding):
    params_path = Path(params_path).expanduser()
    params_bytes = params_path.read_bytes()
    task_config = build_task_config(config, reference_task)
    env = make_task_env(reference_task, task_config, graph_padding)
    network = make_actor_critic(task_config, env)
    template_params = network.init(
        jax.random.PRNGKey(0), env.graph_observation_template()
    )
    try:
        return serialization.from_bytes(template_params, params_bytes)
    except ValueError as exc:
        raise ValueError(
            "failed to load params into the validation network template; "
            "make sure the YAML model settings match the checkpoint architecture"
        ) from exc
