import subprocess
import itertools
import re
import numpy as np

# Parameter ranges
base_spread_values = np.round(np.arange(1.0, 2.05, 0.1), 2)
inventory_factor_values = np.round(np.arange(-0.1, -0.74, -0.05), 2)  # Just -0.1 if step > range?
momentum_factor_values = np.round(np.arange(0.05, 0.66, 0.05), 2)

STRATEGY_PATH = "round4/round4.py"  # assuming this is what you’re testing

def modify_strategy_file(base_spread, inventory_factor, momentum_factor):
    with open(STRATEGY_PATH, "r") as file:
        content = file.read()

    # Regex fix to capture +/- numbers (including decimals)
    content = re.sub(
        r'(self\.base_spreads\s*=\s*{[^}]*?"CROISSANTS"\s*:\s*)[-+]?\d*\.?\d+',
        lambda m: f'{m.group(1)}{base_spread}',
        content,
        count=1
    )

    content = re.sub(
        r'(self\.inventory_factor\s*=\s*{[^}]*?"CROISSANTS"\s*:\s*)[-+]?\d*\.?\d+',
        lambda m: f'{m.group(1)}{inventory_factor}',
        content,
        count=1
    )

    content = re.sub(
        r'(self\.momentum_factor\s*=\s*{[^}]*?"CROISSANTS"\s*:\s*)[-+]?\d*\.?\d+',
        lambda m: f'{m.group(1)}{momentum_factor}',
        content,
        count=1
    )

    with open(STRATEGY_PATH, "w") as file:
        file.write(content)

def run_backtest():
    try:
        result = subprocess.run(["prosperity3bt", STRATEGY_PATH, "6-4"], capture_output=True, text=True, timeout=60)
        output = result.stdout
        match = re.search(r"Total profit:\s*([-\d,]+)", output)
        if match:
            return int(match.group(1).replace(",", ""))
        else:
            print("⚠️ Could not parse profit from output:")
            print(output)
            return None
    except Exception as e:
        print(f"⚠️ Error running backtest: {e}")
        return None

def main():
    results = []
    total_tests = len(base_spread_values) * len(inventory_factor_values) * len(momentum_factor_values)
    print(f"🧪 Total parameter combinations: {total_tests}")
    test_num = 0

    for base_spread, inventory_factor, momentum_factor in itertools.product(
        base_spread_values, inventory_factor_values, momentum_factor_values
    ):
        test_num += 1
        print(f"[{test_num}/{total_tests}] Testing: spread={base_spread}, inventory={inventory_factor}, momentum={momentum_factor}")
        modify_strategy_file(base_spread, inventory_factor, momentum_factor)
        profit = run_backtest()

        if profit is not None:
            results.append({
                "base_spread": base_spread,
                "inventory_factor": inventory_factor,
                "momentum_factor": momentum_factor,
                "profit": profit
            })
            print(f"Profit: {profit}")
        else:
            print(" ❌ Backtest failed or profit missing")

    results.sort(key=lambda x: x["profit"], reverse=True)

    print("\n🏆 Top 5 parameter sets:")
    for r in results[:5]:
        print(f"→ {r}")

if __name__ == "__main__":
    main()
