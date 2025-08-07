This code implements a simple game environment and a fully connected feedforward neural network for deep Q-learning. The agent learns through experience replay to maximize its cumulative reward by interacting within a 10x10 grid, facing enemies and making decisions based on six possible actions.
Overview
    • Game Environment: A grid world where the player must defeat randomly placed enemies. The environment encodes the positions and health of the player and a single active enemy into a compact state vector.
    • Neural Network: A three-layer (input, hidden, output) neural net predicts the Q-value of each possible action, guiding the agent's choices.
    • Learning Algorithm: Q-Learning with experience replay. Experience tuples are stored and used to train the network via stochastic gradient descent.

Neural Network Architecture
Layer
Nodes
Description
Activation
Input
6
Player x/y, Player HP, Enemy x/y, Enemy HP (normalized)
-
Hidden
10
Features extracted from state
Sigmoid
Output
6
Q-values for: move [up, down, left, right], attack, no-op
(Linear)
    • Input vector represents the current state: player's x/y position and health, and single enemy's x/y position and health, all normalized.
    • Hidden layer employs the sigmoid activation function to introduce non-linearity.
    • Output layer produces Q-values (no activation)—one per possible action. The action with the highest Q-value is typically chosen, unless exploring.

State Representation
State input vector for the neural net (size 6):
    1. Player X position / grid size
    2. Player Y position / grid size
    3. Player HP / 100.0
    4. Enemy X position / grid size
    5. Enemy Y position / grid size
    6. Enemy HP / 50.0
If no enemy is alive, enemy values are set to zero.

Action Space
    • Actions [0–3]: Move up, down, left, right (with grid boundary checks).
    • Action 4: Attack if adjacent to enemy (reduces enemy HP, rewards for hit/kill).
    • Action 5: No operation ("do nothing"), with a small penalty.
Key Components
    • Weights & Biases: Stored as 2D arrays with small random initialization.
    • Forward Pass: Computes hidden activations and output Q-values given a current state.
    • Action Selection: Epsilon-greedy strategy: mostly pick max-Q action, occasionally pick random for exploration.
    • Experience Replay: Stores fixed-size buffer of (state, action, reward, next_state, done) tuples for stable, efficient learning.
    • Training: Samples random batches from replay memory, computes TD error, and applies gradient updates on both layers (vanilla backprop due to shallow architecture).
Game Mechanics
    • Player/enemy positions tracked on a 10x10 ASCII grid.
    • Each episode: Game initializes with the player at center, one random active enemy at its fixed position.
    • End of episode: When player dies or all enemies defeated.
    • Rewards: Attacking/killing enemies yields reward, being hit or stalling incurs penalties.
Visualization & Logging
    • ASCII grid shows real-time placement of player and (single) enemy in color.
    • Reward logging: Each episode's total reward is recorded to CSV and can be visualized as ASCII bar chart and improvement summary in the terminal.
Important Parameters
    • GAMMA: Reward discount factor (0.95)
    • EPSILON: Exploration probability (0.1)
    • LEARNING_RATE: Update rate for backprop (0.01)
    • MEMORY_SIZE: Size of experience replay buffer (1000)
    • BATCH_SIZE: Size of training batch (32)
    • GRID_SIZE: Game map dimensions (10x10)

Entry Point
The program's main() function orchestrates:
    • Neural net initialization
    • User input for visualization/delay
    • Main training loop over episodes:
        ◦ Runs episodes, stores experiences, trains, logs progress
        ◦ Optionally visualizes each episode and prints ASCII learning curve
Core Functions Reference
    • init_network(): Initialize neural net weights/biases
    • forward(state, out_values, hidden): Forward-pass computation
    • choose_action(state): Epsilon-greedy action selection
    • store_experience(state, action, reward, next_state, done): Adds sample to replay buffer
    • train_network(): Performs Q-learning update on a randomized sample from replay memory
    • init_game(): Initializes the game environment
    • get_game_state(state): Encodes current environment into state vector
    • take_action(action): Updates environment for agent and enemy actions, applies rewards/penalties
    • is_game_over(): Checks end condition (player dead or enemy dead)
    • print_grid(), ascii_bar_chart(), print_improvement_summary(): Visualization & monitoring
Customization
You may adjust parameters (e.g., number of enemies, actions, architecture size) via the #define directives. The environment and reward structure can be expanded for more sophisticated scenarios.
