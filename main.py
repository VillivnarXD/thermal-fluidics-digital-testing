import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# simulation settings and constants
CHIP_SIZE = 30
TOTAL_CYCLES = 200
ALPHA = 0.12          # thermal diffusivity of silicon
PASSIVE_LOSS = 0.02   # passive heat loss
FLUID_COOLING = 0.45  # fluid cooling capacity
LOOKAHEAD_STEPS = 10  # prediction lookahead
MAX_SAFE_TEMP = 60.0  # max allowed temp

# core coordinates
CORE_ROWS = slice(11, 19)
CORE_COLS = slice(11, 19)

# generate training data
print("Generating synthetic workload history for ML training...")
temp_grid = np.full((CHIP_SIZE, CHIP_SIZE), 25.0)
workloads = []
temp_history = []

# fix seed for same results
seed = np.random.seed(42)

# create workload spikes
for t in range(TOTAL_CYCLES + LOOKAHEAD_STEPS + 5):
    q_step = np.zeros((CHIP_SIZE, CHIP_SIZE))
    if 10 <= t <= 150:
        q_step[CORE_ROWS, CORE_COLS] = np.random.uniform(4.0, 8.5)
    workloads.append(q_step)

# run baseline sim to get data
for t in range(TOTAL_CYCLES + LOOKAHEAD_STEPS):
    temp_history.append(temp_grid.copy())
    laplacian = np.zeros_like(temp_grid)
    
    # 2d heat equation laplacian
    for row in range(1, CHIP_SIZE - 1):
        for col in range(1, CHIP_SIZE - 1):
            laplacian[row, col] = (temp_grid[row+1, col] + temp_grid[row-1, col] + 
                               temp_grid[row, col+1] + temp_grid[row, col-1] - 
                               4 * temp_grid[row, col])
            
    next_grid = temp_grid + (ALPHA * laplacian) - (PASSIVE_LOSS * (temp_grid - 25.0)) + workloads[t]
    temp_grid = np.clip(next_grid, 25.0, 150.0)

# train model
print("Formatting datasets and training local Ridge model...")
X, y = [], []

# extract features from core area
for t in range(1, TOTAL_CYCLES - LOOKAHEAD_STEPS):
    for row in range(11, 19):
        for col in range(11, 19):
            curr_t = temp_history[t][row, col]
            momentum = curr_t - temp_history[t-1][row, col] # thermal momentum
            future_load = workloads[t + LOOKAHEAD_STEPS][row, col]
            
            X.append([curr_t, momentum, future_load])
            y.append(temp_history[t + LOOKAHEAD_STEPS][row, col])

X, y = np.array(X), np.array(y)
thermal_model = Ridge(alpha=1.0)
thermal_model.fit(X, y)

# check r2 score
preds = thermal_model.predict(X)
print(f"Model Training Complete. R2 Accuracy Score: {r2_score(y, preds):.4f}")

# cooling simulations
print("Evaluating cooling scenarios...")

def run_simulation(mode="uncooled"):
    T = np.full((CHIP_SIZE, CHIP_SIZE), 25.0)
    T_prev = T.copy()
    peak_temps = []
    
    for t in range(TOTAL_CYCLES):
        valves = np.zeros((CHIP_SIZE, CHIP_SIZE))
        curr_load = workloads[t]
        
        if mode == "reactive":
            # reactive method triggers over max safe temp
            valves[T > MAX_SAFE_TEMP] = 1.0
            
        elif mode == "predictive_ai":
            # ai method predicts future temp
            for row in range(1, CHIP_SIZE - 1):
                for col in range(1, CHIP_SIZE - 1):
                    t_curr = T[row, col]
                    t_mom = t_curr - T_prev[row, col]
                    w_fut = workloads[min(t + LOOKAHEAD_STEPS, TOTAL_CYCLES-1)][row, col]
                    
                    pred_t = thermal_model.predict([[t_curr, t_mom, w_fut]])[0]
                    if pred_t > MAX_SAFE_TEMP:
                        valves[row, col] = 1.0
                        
        # physics update loop
        T_prev = T.copy()
        T_next = T.copy()
        for row in range(1, CHIP_SIZE - 1):
            for col in range(1, CHIP_SIZE - 1):
                laplacian = T[row+1, col] + T[row-1, col] + T[row, col+1] + T[row, col-1] - 4*T[row, col]
                loss_rate = PASSIVE_LOSS + (FLUID_COOLING * valves[row, col])
                T_next[row, col] = T[row, col] + (ALPHA * laplacian) - (loss_rate * (T[row, col] - 25.0)) + curr_load[row, col]
                
        T = np.clip(T_next, 25.0, 150.0)
        peak_temps.append(T.max())
        
    return peak_temps

# run scenarios
uncooled_log = run_simulation("uncooled")
reactive_log = run_simulation("reactive")
ai_predictive_log = run_simulation("predictive_ai")

# plot results
print("Generating final plot...")
plt.figure(figsize=(9, 5.5))

plt.plot(uncooled_log, label='No Active Cooling', color='#e74c3c', linestyle=':', linewidth=1.8)
plt.plot(reactive_log, label='Standard Reactive Cooling', color='#f39c12', linestyle='--', linewidth=2.0)
plt.plot(ai_predictive_log, label='Predictive AI Cooling', color='#2c3e50', linewidth=2.5)

plt.axhline(y=MAX_SAFE_TEMP, color='black', linestyle='-.', alpha=0.5, label='Target Threshold (60°C)')
plt.title(f"CPU Core Temp Over Time: Reactive vs. Predictive AI Cooling, Seed {seed}", 
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Clock Cycles", fontsize=10)
plt.ylabel("Peak Core Temperature (°C)", fontsize=10)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig('predictive_ai_thermal_results.png', dpi=300)
print("Graph saved successfully as 'predictive_ai_thermal_results.png'")
plt.show()