//Descriptions:
//Each "O" is a neuron.
//Input layer: 6 nodes take the current game state.
//Hidden layer: 10 nodes, with sigmoid activations.
//Output layer: 6 nodes, producing Q-values for possible actions.

//Input Layer    Hidden Layer     Output Layer
//   6 nodes        10 nodes         6 nodes
//   (state)
//      |                |               |
//      |    ________    |               |
//      |   /        \   |               |
//O O O O O O          O O O O O O O O O O
//| | | | | |         /| | | | | | | | | |\
//| | | | | |        / | | | | | | | | | | \
//| | | | | |_______/  | | | | | | | | | |  \
//| | | | | /         /| | | | | | | | | |   |
//| | | | |/         / | | | | | | | | | |   |
//O O O O O O O O O O  | | | | | | | | | |   |
//         \         /  | | | | | | | | | |   |
//          \_______/   | | | | | | | | | |   |
//             \      / | | | | | | | | | |   |
//              \____/  | | | | | | | | | |   |
//                 |    | | | | | | | | | |   |
//                 |    O O O O O O O O O O   |
//                 |        \        /        |
//                        O O O O O O         |
//                       (Q-values/actions)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// Terminal color escape codes for visual clarity in output
#define COLOR_RESET   "\x1b[0m"     // Reset terminal colors
#define COLOR_GREEN   "\x1b[32m"    // Player color: green
#define COLOR_RED     "\x1b[31m"    // Enemy color: red
#define COLOR_WHITE   "\x1b[37m"    // Grid color: white

// Neural network and game parameters definitions
#define INPUT_SIZE 6          // Input vector length: player + one enemy info
#define HIDDEN_SIZE 10        // Number of neurons in hidden layer
#define OUTPUT_SIZE 6         // Number of actions neural net can choose (0-3 move; 4 attack; 5 no-op)
#define MEMORY_SIZE 1000      // Size of replay memory buffer
#define BATCH_SIZE 32         // Number of samples per training batch
#define GAMMA 0.95            // Discount factor for future rewards in Q-learning
#define EPSILON 0.1           // Exploration rate for epsilon-greedy policy
#define LEARNING_RATE 0.01    // Step size for gradient updates
#define GRID_SIZE 10          // Game grid size (10x10)
#define NUM_ENEMIES 5         // Total different enemies defined
#define MAX_BARS 80           // Maximum width of ASCII bar chart

// Enemy struct to represent each enemy's state on grid
typedef struct {
    int x, y;                 // Enemy position (grid coordinates)
    double hp;                // Enemy health points
    int alive;                // Alive flag: 1 if alive, 0 if not
} Enemy;

// Game state holds player's position and HP as well as enemies
typedef struct {
    int player_x, player_y;   // Player position
    double player_hp;         // Player health points
    Enemy enemies[NUM_ENEMIES]; // Array of enemies
} GameState;

// Experience tuple for experience replay memory
typedef struct {
    double state[INPUT_SIZE];      // Current observed state
    int action;                   // Action taken
    double reward;                // Reward received
    double next_state[INPUT_SIZE];// Next observed state after action
    int done;                     // Flag if episode ended
} Experience;

// Neural network parameters: weights and biases
double w1[INPUT_SIZE][HIDDEN_SIZE];  // Input to hidden weights
double w2[HIDDEN_SIZE][OUTPUT_SIZE]; // Hidden to output weights
double b1[HIDDEN_SIZE];               // Hidden layer biases
double b2[OUTPUT_SIZE];               // Output layer biases

Experience replay_memory[MEMORY_SIZE]; // Circular experience replay buffer
int memory_count = 0;                  // Count of stored experiences

GameState game;                       // Global game state variable

// Utility functions for minimum and maximum of two integers
int min(int a, int b) { return (a < b) ? a : b; }
int max(int a, int b) { return (a > b) ? a : b; }

// Sigmoid activation function used in hidden layer neurons
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Initialize all neural network weights and biases to small random values
void init_network() {
    srand((unsigned int)time(NULL));  // Seed the random number generator
    // Input to hidden weights initialization [-0.1, 0.1]
    for(int i=0; i < INPUT_SIZE; i++)
        for(int j=0; j < HIDDEN_SIZE; j++)
            w1[i][j] = ((double)rand() / RAND_MAX)*0.2 - 0.1;
    // Hidden to output weights initialization [-0.1, 0.1]
    for(int i=0; i < HIDDEN_SIZE; i++)
        for(int j=0; j < OUTPUT_SIZE; j++)
            w2[i][j] = ((double)rand() / RAND_MAX)*0.2 - 0.1;
    // Hidden layer biases initialization
    for(int i=0; i < HIDDEN_SIZE; i++)
        b1[i] = ((double)rand() / RAND_MAX)*0.2 - 0.1;
    // Output layer biases initialization
    for(int i=0; i < OUTPUT_SIZE; i++)
        b2[i] = ((double)rand() / RAND_MAX)*0.2 - 0.1;
}

// Forward pass of the neural network: compute Q-values for all actions
// Inputs:
//    state: current state vector normalized between 0 and 1
// Outputs:
//    out_values: raw Q-values of size OUTPUT_SIZE (no activation)
//    hidden: hidden layer post-activation values (sigmoid)
void forward(double* state, double* out_values, double* hidden) {
    // Compute hidden layer activations with sigmoid
    for(int i=0; i < HIDDEN_SIZE; i++) {
        hidden[i] = b1[i];                // Start with bias
        for(int j=0; j < INPUT_SIZE; j++)
            hidden[i] += state[j] * w1[j][i];  // Weighted sum from input
        hidden[i] = sigmoid(hidden[i]);  // Activation function
    }
    // Compute output layer Q-values (linear activation)
    for(int i=0; i < OUTPUT_SIZE; i++) {
        out_values[i] = b2[i];             // Start with bias
        for(int j=0; j < HIDDEN_SIZE; j++)
            out_values[i] += hidden[j] * w2[j][i];
    }
}

// Epsilon-greedy policy chooses an action:
// With probability EPSILON return random action (exploration),
// otherwise return action with maximum Q-value (exploitation).
int choose_action(double* state) {
    double out_values[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    forward(state, out_values, hidden);

    double r = (double)rand() / RAND_MAX;
    if(r < EPSILON)
        return rand() % OUTPUT_SIZE;  // Exploration
    else {
        int max_idx = 0;
        for(int i=1; i < OUTPUT_SIZE; i++) {
            if(out_values[i] > out_values[max_idx])
                max_idx = i;  // Exploitation: best action
        }
        return max_idx;
    }
}

// Store experience tuple for replay in circular buffer
void store_experience(double* state, int action, double reward, double* next_state, int done) {
    int idx = memory_count % MEMORY_SIZE;  // Overwrite oldest if full
    for(int i=0; i < INPUT_SIZE; i++) {
        replay_memory[idx].state[i] = state[i];
        replay_memory[idx].next_state[i] = next_state[i];
    }
    replay_memory[idx].action = action;
    replay_memory[idx].reward = reward;
    replay_memory[idx].done = done;
    memory_count++;
}

// Train the neural network by sampling a batch of experiences from replay memory
// and performing a simple one-step Q-learning update using gradient descent.
void train_network() {
    if(memory_count < BATCH_SIZE) return;  // Not enough data yet

    int batch_size = min(memory_count, BATCH_SIZE);
    int max_samples = min(memory_count, MEMORY_SIZE);

    for(int b=0; b < batch_size; b++) {
        int idx = rand() % max_samples;  // Random sample from memory

        double* state = replay_memory[idx].state;
        double* next_state = replay_memory[idx].next_state;
        int action = replay_memory[idx].action;
        double reward = replay_memory[idx].reward;
        int done = replay_memory[idx].done;

        double q_values[OUTPUT_SIZE], next_q_values[OUTPUT_SIZE];
        double hidden[HIDDEN_SIZE], next_hidden[HIDDEN_SIZE];

        // Predict Q(s,a)
        forward(state, q_values, hidden);
        // Predict Q(s',a') for next state
        forward(next_state, next_q_values, next_hidden);

        // Calculate target using Bellman equation
        double target = reward;
        if(!done) {
            double max_next_q = next_q_values[0];
            for(int i=1; i < OUTPUT_SIZE; i++)
                if(next_q_values[i] > max_next_q)
                    max_next_q = next_q_values[i];
            target += GAMMA * max_next_q;
        }

        // TD error for Q-value update
        double error = target - q_values[action];

        // Update output weights and biases (gradient descent step)
        for(int i=0; i < HIDDEN_SIZE; i++)
            w2[i][action] += LEARNING_RATE * error * hidden[i];
        b2[action] += LEARNING_RATE * error;

        // Backpropagate error to hidden layer neurons (sigmoid derivative)
        double delta_hidden[HIDDEN_SIZE];
        for(int i=0; i < HIDDEN_SIZE; i++)
            delta_hidden[i] = w2[i][action] * error * hidden[i] * (1 - hidden[i]);

        // Update input-hidden weights and biases
        for(int i=0; i < INPUT_SIZE; i++)
            for(int j=0; j < HIDDEN_SIZE; j++)
                w1[i][j] += LEARNING_RATE * delta_hidden[j] * state[i];
        for(int i=0; i < HIDDEN_SIZE; i++)
            b1[i] += LEARNING_RATE * delta_hidden[i];
}
}
// Predefined fixed positions for 5 different enemies
int enemy_positions[NUM_ENEMIES][2] = {
    {8, 8}, {1, 9}, {3, 3}, {7, 2}, {0, 0}
};

// Initialize game state at episode start:
// Player positioned at center with full health
// Exactly one enemy randomly selected, placed and marked alive
void init_game() {
    game.player_x = 5;
    game.player_y = 5;
    game.player_hp = 100.0;
    // Set all enemies as dead/inactive initially
    for(int i=0; i < NUM_ENEMIES; i++)
        game.enemies[i].alive = 0;
    // Pick one enemy randomly and initialize
    int active_idx = rand() % NUM_ENEMIES;
    game.enemies[active_idx].x = enemy_positions[active_idx][0];
    game.enemies[active_idx].y = enemy_positions[active_idx][1];
    game.enemies[active_idx].hp = 50.0;
    game.enemies[active_idx].alive = 1;
}

// Update state array for the neural net input:
// Normalized player position, HP; and single active enemy position, HP
void get_game_state(double* state) {
    state[0] = (double)game.player_x / GRID_SIZE;
    state[1] = (double)game.player_y / GRID_SIZE;
    state[2] = game.player_hp / 100.0;

    // Find active enemy index (only one is alive)
    int active_idx = -1;
    for(int i=0; i < NUM_ENEMIES; i++)
        if(game.enemies[i].alive) {
            active_idx = i;
            break;
        }

    if(active_idx == -1) {
        // No active enemy - zero vector
        state[3] = 0;
        state[4] = 0;
        state[5] = 0;
    } else {
        // Enemy position and health normalized
        state[3] = (double)game.enemies[active_idx].x / GRID_SIZE;
        state[4] = (double)game.enemies[active_idx].y / GRID_SIZE;
        state[5] = game.enemies[active_idx].hp / 50.0;
    }
}

// Execute chosen action and update game state. Return immediate reward obtained.
double take_action(int action) {
    double reward = -0.1;  // Small negative reward each step to incentivize efficiency

    int old_x = game.player_x;
    int old_y = game.player_y;

    // Find active enemy
    int active_idx = -1;
    for(int i=0; i < NUM_ENEMIES; i++)
        if(game.enemies[i].alive) {
            active_idx = i;
            break;
        }

    switch(action) {
        case 0:  // Move up
            game.player_y = max(0, game.player_y - 1);
            break;
        case 1:  // Move down
            game.player_y = min(GRID_SIZE -1, game.player_y + 1);
            break;
        case 2:  // Move left
            game.player_x = max(0, game.player_x - 1);
            break;
        case 3:  // Move right
            game.player_x = min(GRID_SIZE -1, game.player_x +1);
            break;
        case 4:  // Attack enemy if adjacent
            if(active_idx != -1 &&
               abs(game.player_x - game.enemies[active_idx].x) <= 1 &&
               abs(game.player_y - game.enemies[active_idx].y) <= 1) {
                game.enemies[active_idx].hp -= 10;
                reward += 5;  // Reward for successful attack
                if(game.enemies[active_idx].hp <= 0) {
                    game.enemies[active_idx].alive = 0;
                    reward += 50;  // Additional reward for defeating enemy
                }
            } else {
                reward -= 1; // Penalty if attack misses (no enemy adjacent)
            }
            break;
        case 5:  // No-op (do nothing) action with slight penalty
            reward -= 0.05;
            break;
        default:
            break;
    }

    // Enemy moves towards the player if alive and attacks when adjacent
    if(active_idx != -1 && game.enemies[active_idx].alive) {
        Enemy *e = &game.enemies[active_idx];
        // Move enemy 1 step closer to player in x and y
        if(e->x < game.player_x) e->x++;
        else if(e->x > game.player_x) e->x--;
        if(e->y < game.player_y) e->y++;
        else if(e->y > game.player_y) e->y--;

        // Enemy attacks player if adjacent
        if(abs(game.player_x - e->x) <= 1 && abs(game.player_y - e->y) <= 1) {
            game.player_hp -= 5;
            reward -= 10;  // Negative reward for player taking damage
        }
    }

    // Penalty for attempting to move but not changing position (blocked by wall)
    if(action <= 3 && game.player_x == old_x && game.player_y == old_y)
        reward -= 2;

    return reward;
}

// Check if episode ended: player death or enemy defeated ends episode
int is_game_over() {
    if(game.player_hp <= 0)
        return 1;   // Player died: end episode
    // Check if any enemy is still alive
    for(int i=0; i < NUM_ENEMIES; i++)
        if(game.enemies[i].alive)
            return 0;   // Enemy alive, game continues
    return 1;           // No enemies alive, episode over
}

// Clears the terminal screen using ANSI escape codes for neat visual updates
void clear_screen() {
    printf("\x1b[H\x1b[2J");
}

// Sleep for given milliseconds (usleep works with microseconds)
void sleep_ms(int ms) {
    usleep(ms * 1000);
}

// Print the current game grid with player and single active enemy positions
// Player represented as green 'P', enemies as red 'E<#>'
void print_grid() {
    clear_screen();

    // Print each grid cell row by row
    for(int y = 0; y < GRID_SIZE; y++) {
        for(int x = 0; x < GRID_SIZE; x++) {
            // Print player if in this cell (green)
            if(game.player_x == x && game.player_y == y) {
                printf(COLOR_GREEN "P " COLOR_RESET);
            } else {
                // Otherwise print enemy if in this cell (red + enemy number)
                int printed = 0;
                for(int i=0; i < NUM_ENEMIES; i++) {
                    if(game.enemies[i].alive && game.enemies[i].x == x && game.enemies[i].y == y) {
                        printf(COLOR_RED "E%d" COLOR_RESET, i+1);
                        printed = 1;
                        break;
                    }
                }
                // Else print empty grid cell as white dot
                if(!printed)
                    printf(COLOR_WHITE ". " COLOR_RESET);
            }
        }
        printf("\n");
    }
    // Print health status below grid
    printf("Player HP: %.1f\nEnemy HP: ", game.player_hp);
    for(int i=0; i < NUM_ENEMIES; i++) {
        if(game.enemies[i].alive)
            printf("E%d:%.1f ", i+1, game.enemies[i].hp);
    }
    printf("\n");
}

// Draw a simple ASCII bar chart of total rewards per episode.
// Skips some episodes if too many, to fit terminal width.
void ascii_bar_chart(double *data, int N, int max_bar_width, int skip) {
    double min_val = data[0], max_val = data[0];
    for(int i=1; i < N; i++) {
        if(data[i] < min_val) min_val = data[i];
        if(data[i] > max_val) max_val = data[i];
    }
    // Add epsilon to avoid division by zero
    if(fabs(max_val - min_val) < 1e-8) max_val += 1.0;

    printf("\nLearning Curve (Total Reward per Episode)\n");
    printf("----------------------------------------\n");
    for(int i=0; i < N; i+=skip) {
        int bar_len = (int)((data[i]-min_val) / (max_val-min_val) * max_bar_width);
        printf("Ep%4d: ", i+1);
        for(int j=0; j < bar_len; j++) putchar('#');
        printf(" (%.2f)\n", data[i]);
    }
    printf("----------------------------------------\n");
    printf("Min: %.2f   Max: %.2f\n\n", min_val, max_val);
}

// Print a summary showing percentage improvement in average total reward
// over slices of length 'interval' compared to the first interval.
void print_improvement_summary(double *data, int N, int interval) {
    if(N < interval) return;

    // Compute average reward of first interval as baseline
    double base_sum = 0.0;
    for(int i=0; i < interval; i++)
        base_sum += data[i];
    double base_avg = base_sum / interval;

    printf("Learning Improvement Summary per %d episodes:\n", interval);
    printf("---------------------------------------------\n");

    // Iterate over intervals for the entire training set
    for(int start = 0; start < N; start += interval) {
        int end = min(start + interval, N);
        double seg_sum = 0.0;
        int count = end - start;
        for(int i = start; i < end; i++)
            seg_sum += data[i];

        double seg_avg = seg_sum / count;
        double improvement = ((seg_avg - base_avg) / fabs(base_avg)) * 100.0;

        // Print average reward and % improvement relative to baseline
        printf("Episodes %4d-%4d: Avg Reward = %.2f, Improvement = %+6.2f%%\n",
               start + 1, end, seg_avg, improvement);
    }
    printf("\n");
}

int main() {
    init_network();                    // Initialize neural network weights and biases
    srand((unsigned int)time(NULL));  // Seed RNG

    int sleep_duration_ms = 100;      // Default delay in ms between steps
    printf("Enter delay in milliseconds between steps (e.g., 100): ");
    if(scanf("%d", &sleep_duration_ms) != 1 || sleep_duration_ms < 0) {
        printf("Invalid input, using default %d ms.\n", sleep_duration_ms);
        sleep_duration_ms = 100;
    }

    // Whether to display each match visually as grid or skip visualization for speed
    int visualize = 1;
    char choice[10];
    printf("Visualize each match? (y/n): ");
    if(scanf("%s", choice) != 1) {
        printf("Invalid input, defaulting to 'yes'.\n");
    } else {
        if(choice[0] == 'n' || choice[0] == 'N')
            visualize = 0;
    }

    // Open file to log total rewards per episode
    FILE* log_file = fopen("learning_log.csv", "w");
    if (!log_file) {
        perror("Could not open log file");
        return 1;
    }
    fprintf(log_file, "episode,total_reward\n");

    const int episodes = 1000;                                            // Total number of episodes
    double *reward_history = malloc(episodes * sizeof(double));           // Store total rewards for visualization
    if(!reward_history) {
        perror("Memory allocation failed");
        fclose(log_file);
        return 1;
    }

    double state[INPUT_SIZE], next_state[INPUT_SIZE];   // Buffers to store current and next states

    // Main training loop over all episodes
    for(int episode = 0; episode < episodes; episode++) {
        init_game();                 // Initialize game state with player and single random enemy
        get_game_state(state);       // Get normalized initial state vector
        double total_reward = 0.0;   // Reset total reward for this episode

        // Run steps until episode ends
        while(1) {
            if(visualize)
                print_grid();        // Show current game grid if user wants visualization

            int action = choose_action(state);      // NN selects next action (eps-greedy)
            double reward = take_action(action);    // Take action, update game, receive reward
            get_game_state(next_state);             // Update next state vector for NN
            int done = is_game_over();               // Check if episode ended

            store_experience(state, action, reward, next_state, done);  // Save experience tuple
            train_network();                        // Perform training iteration on a batch

            total_reward += reward;                 // Accumulate episode total reward

            // Update current state to next state
            for(int i=0; i < INPUT_SIZE; i++)
                state[i] = next_state[i];

            sleep_ms(sleep_duration_ms);            // Pause step according to user input speed

            if(done) {
                if(visualize)
                    print_grid();                    // Show final grid of episode

                // Print and log total reward this episode
                printf("Episode %d ended. Total Reward: %.2f\n", episode + 1, total_reward);
                fprintf(log_file, "%d,%.2f\n", episode + 1, total_reward);
                reward_history[episode] = total_reward;

                sleep_ms(1000);                     // Pause briefly before next episode
                break;
            }
        }
    }

    fclose(log_file);                             // Close the CSV log file

    // If many episodes, skip some in the chart to fit terminal
    int skip = 1;
    if(episodes > 80)
        skip = episodes / 80;

    // Visualize learning curve as ASCII bar chart in terminal
    ascii_bar_chart(reward_history, episodes, MAX_BARS, skip);

    // Print progress summary showing learning improvements every 100 episodes
    print_improvement_summary(reward_history, episodes, 100);

    free(reward_history);   // Free allocated memory

    return 0;
}
