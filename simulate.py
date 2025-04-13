import subprocess
import re
import itertools

# Path to your trading strategy file
strategy_file = "/Users/ethansavar/Documents/IMCProsperity2025/round2.py"

# Read original strategy content
with open(strategy_file, "r") as f:
    original_code = f.read()

# Define search patterns for base spread and sensitivity
base_pattern = re.compile(r"base_spread\s*=\s*\d+")
sensitivity_pattern = re.compile(r"spread_sensitivity\s*=\s*\d+")

# Define the parameter ranges
base_spread_range = range(15, 21)  # 5 to 20
spread_sensitivity_range = range(4, 20)  # 3 to 25

# Store results
results = []

for base_spread, spread_sensitivity in itertools.product(base_spread_range, spread_sensitivity_range):
    # Modify the code with new parameters
    modified_code = base_pattern.sub(f"base_spread = {base_spread}", original_code)
    modified_code = sensitivity_pattern.sub(f"spread_sensitivity = {spread_sensitivity}", modified_code)

    # Save modified code
    with open(strategy_file, "w") as f:
        f.write(modified_code)

    # Run the simulator
    result = subprocess.run(
        ["prosperity3bt", "round2.py", "2"],
        capture_output=True,
        text=True
    )

    output = result.stdout

    # Extract JAMS profit or total profit
    match = re.search(r"JAMS:\s*(-?[\d,]+)", output)
    if match:
        jams_profit = int(match.group(1).replace(",", ""))
        results.append((jams_profit, base_spread, spread_sensitivity))
        print(f"Tested base_spread={base_spread}, spread_sensitivity={spread_sensitivity}, JAMS Profit={jams_profit}")
    else:
        print(f"Failed to get profit for base_spread={base_spread}, spread_sensitivity={spread_sensitivity}")
        continue

# Restore original file
with open(strategy_file, "w") as f:
    f.write(original_code)

# Sort and print the best result
best = max(results, key=lambda x: x[0])
print(f"\nBest combo: base_spread={best[1]}, spread_sensitivity={best[2]}, JAMS Profit={best[0]}")
