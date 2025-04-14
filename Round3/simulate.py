import subprocess
import itertools
import re
import numpy as np

# Generate parameter ranges
base_spread_values = np.round(np.arange(1.0, 2.05, 0.1), 2)
inventory_factor_values = np.round(np.arange(-0.6, -0.049, 0.05), 2)
momentum_factor_values = np.round(np.arange(0.1, 0.76, 0.05), 2)

def modify_strategy_file(base_spread, inventory_factor, momentum_factor):
    path = "round3/round3.py"
    with open(path, "r") as file:
        content = file.read()

    # Base spread (inside self.base_spreads = { ... })
    content = re.sub(
        r'(self\.base_spreads\s*=\s*{[^}]*?"SQUID_INK":\s*)[0-9.]+',
        lambda m: f'{m.group(1)}{base_spread}',
        content,
        count=1
    )

    # Inventory factor (inside self.inventory_factor = { ... })
    content = re.sub(
        r'(self\.inventory_factor\s*=\s*{[^}]*?"SQUID_INK":\s*)-?[0-9.]+',
        lambda m: f'{m.group(1)}{inventory_factor}',
        content,
        count=1
    )

    # Momentum factor (inside self.momentum_factor = { ... })
    content = re.sub(
        r'(self\.momentum_factor\s*=\s*{[^}]*?"SQUID_INK":\s*)[0-9.]+',
        lambda m: f'{m.group(1)}{momentum_factor}',
        content,
        count=1
    )

    with open(path, "w") as file:
        file.write(content)



def run_backtest():
    result = subprocess.run(["prosperity3bt", "round3/round3.py", "6-2"], capture_output=True, text=True)
    output = result.stdout
    match = re.search(r"Total profit:\s*([\d,\-]+)", output)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

def main():
    results = []
    total_tests = len(base_spread_values) * len(inventory_factor_values) * len(momentum_factor_values)
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
            print(f" -> Profit: {profit}")
        else:
            print(" -> Failed to parse profit")

    results.sort(key=lambda x: x["profit"], reverse=True)

    print("\n🏆 Top 5 parameter sets:")
    for r in results[:5]:
        print(f"{r}")

if __name__ == "__main__":
    main()
