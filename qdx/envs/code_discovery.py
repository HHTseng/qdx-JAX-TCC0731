from functools import lru_cache
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional
from itertools import combinations, product
from math import comb
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
        
        # Initialize action tensor
        self.actions = self.action_matrix()
        
        # Symplectic metric Omega
        self.Omega = self._cached_omega(self.n_qubits_physical)
        
        # Initialize error operators and probabilities
        self.E_mu, self.p_mu = self.error_operators()
        
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
        max_softness = min(int(softness), int(num_stabilizers))
        soft_elements = sum(
            comb(num_stabilizers, weight)
            for weight in range(1, max_softness + 1)
        )
        s_struct = np.zeros((soft_elements, num_stabilizers), dtype=np.uint8)

        start_idx = 0
        for weight in range(1, max_softness + 1):
            for row_offset, indices in enumerate(
                combinations(range(num_stabilizers), weight)
            ):
                s_struct[start_idx + row_offset, indices] = 1
            start_idx += comb(num_stabilizers, weight)

        assert np.prod(np.any(s_struct, axis=1)), "There is a row with all zeroes"
        return jnp.asarray(s_struct)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_error_operators(n_qubits_physical, code_distance, p_identity):
        max_weight = min(int(code_distance) - 1, int(n_qubits_physical))
        total_errors = sum(
            comb(n_qubits_physical, weight) * (3 ** weight)
            for weight in range(1, max_weight + 1)
        )
        error_ops = np.zeros(
            (total_errors, 2 * n_qubits_physical), dtype=np.uint8
        )
        probabilities = np.empty((total_errors,), dtype=np.float32)

        p_single = np.float32((1.0 - p_identity) / 3.0)
        p_identity = np.float32(p_identity)
        row = 0
        for weight in range(1, max_weight + 1):
            weight_probability = np.float32(
                (p_single ** weight) * (p_identity ** (n_qubits_physical - weight))
            )
            for positions in combinations(range(n_qubits_physical), weight):
                for pauli_types in product((1, 2, 3), repeat=weight):
                    row_values = error_ops[row]
                    for position, pauli_type in zip(positions, pauli_types):
                        if pauli_type != 3:
                            row_values[position] = 1
                        if pauli_type != 1:
                            row_values[n_qubits_physical + position] = 1
                    probabilities[row] = weight_probability
                    row += 1

        return jnp.asarray(error_ops), jnp.asarray(probabilities)
    
    def stabilizer_elements(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (self.S_struct @ tableau) % 2

    
    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> chex.Array:
        
        action_matrix = []
        self.action_string = []
        self.action_string_stim = []

        for gate in self.gates:
            ## One qubit gate
            if len(signature(gate).parameters) == 1:
                for n_qubit in range(self.n_qubits_physical):                    
                    action_matrix.append(gate(n_qubit))
                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))
                    

            ## Two qubit gates
            elif len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1]))                    
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))

                    
        return jnp.array(action_matrix, dtype=jnp.uint8)
    
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
        
        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        self.num_KL = len(self.E_mu) - jnp.sum(jnp.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, axis=1), axis=0) - jnp.sum(inS)
        
        # Return the weighted KL sum rescaled by lbda
        return self.lbda * ( jnp.sum(self.p_mu) - jnp.sum(self.p_mu * jnp.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, axis=1), axis=0) - jnp.dot(self.p_mu, jnp.sum(inS, axis=-1)) )
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        
        prev_terminal = self.is_terminal(state, params)
        
        # Update state
        state = EnvState( (state.tableau @ self.actions[action]) % 2, state.time + 1)
        
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
            tableau = init_state,
            time = 0
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
                "time": spaces.Discrete(params.max_steps),
            }
        )
