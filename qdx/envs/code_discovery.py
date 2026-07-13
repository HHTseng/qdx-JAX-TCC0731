from functools import lru_cache
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Tuple, Optional
from itertools import combinations, product
from math import comb
from qdx.action_space import ACTION_SPACE_VERSION, build_action_specs
from qdx.runtime_cache import (
    build_error_operators_upto,
    build_s_structure,
    load_or_build_array_bundle,
)
from qdx.simulators import TableauSimulator
import numpy as np
from jax import lax


@struct.dataclass
class EnvState:
    """
    This class will contain the state of the environment:

    tableau: binary array of size 2*n_qubits_physical*(n_qubits_physical-n_qubits_logical)
    time: integer from 0 to max_steps
    """
    tableau: jnp.array
    time: int
    pending_action_mask: jnp.array

# We ignore this class   
@struct.dataclass
class EnvParams:
    n: int = 7
    k: int = 1
    d: int = 3
    max_steps_in_episode: int = 20
    
    
class CodeDiscovery(environment.Environment):
    """
    Environment for the automatic discovery of QEC codes and encodings.
    
    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        code_distance (int): Target code distance
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        pI (float, optional): Probability of no error in the noise channel. Default: 0.9
        softness (int, optional): Parameter that controls the size of the stabilizer subgroup to be generated. Default: 1
    """
    _ACTION_MATRIX_MEMORY_CACHE = {}

    def __init__(self,
            n_qubits_physical,
            n_qubits_logical,
            code_distance,
            gates,
            graph=None,
            max_steps = 30,
            lbda = 100,
            pI=0.9,
            softness=1,
                ):
        super().__init__()
        
        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.pI = pI # Probability of no error
        
        
        self.graph = graph
        if self.graph is None:
            self.graph = []
            # Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))
                    

        
        self.obs_shape = (2 * n_qubits_physical * (n_qubits_physical - n_qubits_logical), )
        self.action_specs = build_action_specs(
            self.n_qubits_physical,
            self.gates,
            self.graph,
        )
        
        # Initialize action tensor and v1.4 action-relation tables
        self.actions = self.action_matrix()
        self._configure_action_relations(self.actions.shape[0])
        
        # Symplectic metric Omega
        self.Omega = self._cached_omega(self.n_qubits_physical)
        
        # Initialize error operators and probabilities
        self.E_mu, self.p_mu = self.error_operators()
        self.E_mu_Omega = self.E_mu @ self.Omega
        
        # Initialize stabilizer group structure
        self.generate_S_structure(softness) # This generates self.S_struct
        
        # Initialize num_KL
        self.num_KL = len(self.E_mu)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()
    
    def generate_S_structure(self, softness):
        # Generate the structure of the stabilizer group (S) based on the softness parameter.

        num = self.n_qubits_physical - self.n_qubits_logical
        self.S_struct = self._cached_s_structure(num, int(softness))

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_omega(n_qubits_physical):
        return jnp.kron(
            jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8),
            jnp.eye(n_qubits_physical, dtype=jnp.uint8),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_s_structure(num_stabilizers, softness):
        arrays = load_or_build_array_bundle(
            "code_discovery_s_structure",
            {
                "num_stabilizers": int(num_stabilizers),
                "softness": int(softness),
            },
            lambda: {
                "s_struct": build_s_structure(num_stabilizers, softness),
            },
        )
        return jnp.asarray(arrays["s_struct"])

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_error_operators(n_qubits_physical, code_distance, p_identity):
        def build():
            error_ops, probabilities = build_error_operators_upto(
                n_qubits_physical, code_distance, p_identity
            )
            return {
                "error_ops": error_ops,
                "probabilities": probabilities,
            }

        arrays = load_or_build_array_bundle(
            "code_discovery_error_operators",
            {
                "n_qubits_physical": int(n_qubits_physical),
                "code_distance": int(code_distance),
                "p_identity": float(p_identity),
            },
            build,
        )
        return (
            jnp.asarray(arrays["error_ops"]),
            jnp.asarray(arrays["probabilities"]),
        )

    def stabilizer_elements(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (self.S_struct @ tableau) % 2

    
    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> chex.Array:
        gate_names = tuple(
            f"{gate.__module__}.{gate.__qualname__}" for gate in self.gates
        )
        graph_edges = tuple((int(src), int(dst)) for src, dst in self.graph)
        memory_key = (
            ACTION_SPACE_VERSION,
            int(self.n_qubits_physical),
            gate_names,
            graph_edges,
        )
        cached = self._ACTION_MATRIX_MEMORY_CACHE.get(memory_key)
        if cached is None:
            action_specs = self.action_specs

            def build():
                action_matrix = []
                action_string = []
                action_string_stim = []

                for spec in action_specs:
                    gate = self.gates[spec.gate_id]
                    args = spec.gate_args()
                    action_matrix.append(gate(*args))
                    if spec.arity == 1:
                        qubit = args[0]
                        action_string.append('%s-%d' % (gate.__name__, qubit))
                        action_string_stim.append(
                            '.%s(%d)' % (gate.__name__.lower(), qubit)
                        )
                    else:
                        first, second = args
                        action_string.append(
                            '%s-%d-%d' % (gate.__name__, first, second)
                        )
                        action_string_stim.append(
                            '.%s(%d, %d)'
                            % (gate.__name__.lower(), first, second)
                        )

                return {
                    "actions": np.asarray(action_matrix, dtype=np.uint8),
                    "action_string": np.asarray(action_string),
                    "action_string_stim": np.asarray(action_string_stim),
                }

            arrays = load_or_build_array_bundle(
                "code_discovery_action_matrix",
                {
                    "action_space_version": ACTION_SPACE_VERSION,
                    "n_qubits_physical": int(self.n_qubits_physical),
                    "gate_names": gate_names,
                    "graph_edges": graph_edges,
                },
                build,
            )
            cached = (
                jnp.asarray(arrays["actions"]),
                tuple(str(value) for value in arrays["action_string"].tolist()),
                tuple(
                    str(value)
                    for value in arrays["action_string_stim"].tolist()
                ),
            )
            self._ACTION_MATRIX_MEMORY_CACHE[memory_key] = cached

        actions, action_string, action_string_stim = cached
        self.action_string = list(action_string)
        self.action_string_stim = list(action_string_stim)
        return actions

    def _configure_action_relations(self, max_actions: Optional[int] = None) -> None:
        action_count = int(self.actions.shape[0])
        max_actions = action_count if max_actions is None else int(max_actions)
        if max_actions < action_count:
            raise ValueError("max_actions cannot be smaller than action count")

        self.max_actions = max_actions
        base_action_mask = np.zeros(max_actions, dtype=bool)
        base_action_mask[:action_count] = True
        self._base_action_mask = jnp.asarray(base_action_mask)

        commute_table, cancel_table = self._build_action_relation_tables(max_actions)
        self._commute_table = jnp.asarray(commute_table)
        self._cancel_table = jnp.asarray(cancel_table)

    def _build_action_relation_tables(self, max_actions: int):
        actions = np.asarray(self.actions, dtype=np.uint16)
        action_count = int(actions.shape[0])
        width = int(actions.shape[-1])
        identity = np.eye(width, dtype=np.uint8)
        commute_table = np.zeros((max_actions, max_actions), dtype=bool)
        cancel_table = np.zeros((max_actions, max_actions), dtype=bool)

        for action_index in range(action_count):
            action_matrix = actions[action_index]
            left_products = np.matmul(action_matrix, actions) % 2
            right_products = np.matmul(actions, action_matrix) % 2
            commute_table[action_index, :action_count] = np.all(
                left_products == right_products, axis=(-2, -1)
            )
            cancel_table[action_index, :action_count] = np.all(
                left_products == identity, axis=(-2, -1)
            )

        return commute_table, cancel_table

    def update_pending_action_mask(self, pending_action_mask, action):
        commutes = self._commute_table[:, action]
        cancels = self._cancel_table[:, action]
        return jnp.where(
            commutes,
            jnp.logical_xor(pending_action_mask, cancels),
            False,
        )

    def dynamic_action_mask(self, state: EnvState):
        return self._base_action_mask & ~state.pending_action_mask

    def get_observation(self, tableau):
        '''
        Extract the check matrix for the observation
        '''
        ## Only generators without sign
        check_mat = tableau[self.n_qubits_physical + self.n_qubits_logical:].astype(jnp.uint8)
     
        return check_mat
    
    def error_operators(self, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        # Build symplectic X/Z bits directly instead of going through Python
        # permutations and Stim PauliString objects for every operator.
        return self._cached_error_operators(
            self.n_qubits_physical,
            self.d,
            float(self.pI),
        )

    
    def check_KL(self, state: EnvState, params: Optional[EnvParams] = EnvParams):
        # Check the Knill-Laflamme conditions for error correction. This is used to reward the agent
        
        # Extract the stabilizer generators
        check_matrix = state.tableau[self.n_qubits_physical + self.n_qubits_logical:]
        
        # Update the stabilizer group S
        S = self.stabilizer_elements(check_matrix)
        
        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        inS = jax.vmap(jnp.logical_xor, in_axes=(None,0))(S, self.E_mu)
        inS = jnp.prod(jnp.logical_not(inS), axis=-1)
        inS_per_error = jnp.sum(inS, axis=-1)

        # Reuse the symplectic product result across the KL count and reward.
        violations = jnp.any((self.E_mu_Omega @ check_matrix.T) % 2, axis=1)

        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        self.num_KL = len(self.E_mu) - jnp.sum(violations, axis=0) - jnp.sum(inS_per_error)

        # Return the weighted KL sum rescaled by lbda
        return self.lbda * (
            jnp.sum(self.p_mu)
            - jnp.sum(self.p_mu * violations, axis=0)
            - jnp.dot(self.p_mu, inS_per_error)
        )
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        
        prev_terminal = self.is_terminal(state, params)
        
        # Update state
        new_pending_action_mask = self.update_pending_action_mask(
            state.pending_action_mask,
            action,
        )
        state = EnvState(
            (state.tableau @ self.actions[action]) % 2,
            state.time + 1,
            new_pending_action_mask,
        )
        
        # Update KLs
        reward = -self.check_KL(state) 

        # Evaluate termination conditions
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        
        tableau = TableauSimulator(self.n_qubits_physical)
        init_state = tableau.current_tableau[0]
        
        state = EnvState(
            tableau=init_state,
            time=0,
            pending_action_mask=jnp.zeros(self.max_actions, dtype=bool),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state."""
        
        return self.get_observation(state.tableau).flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = self.num_KL == 0 # self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CodeDiscovery"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """Number of actions possible in environment."""
        return self.actions.shape[0]

    def action_space(
        self, params: Optional[EnvParams] = EnvParams
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""

        return spaces.Box(0, 1, self.obs_shape, dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, jnp.uint8),
                "time": spaces.Discrete(self.max_steps),
                "pending_action_mask": spaces.Box(
                    0, 1, (self.max_actions,), dtype=jnp.bool_
                ),
            }
        )
