# Define the container information as a list of tuples.
# Each tuple is (treasure_multiplier, inhabitants_already_choosing)
containers = [
    (10, 1),
    (80, 6),
    (37, 3),
    (90, 10),
    (31, 2),
    (17, 1),
    (50, 4),
    (20, 2),
    (73, 4),
    (89, 8)
]

# Base treasure value
base_treasure = 10000

# Total inhabitants from other players
total_inhabitants = sum(inh for _, inh in containers)  # For our data: 41

def compute_profit(multiplier, inhabitants, total_opens):
    new_count = inhabitants + 1
    division_factor = new_count + (new_count / total_opens) * 100
    profit = (base_treasure * multiplier) / division_factor
    return profit

# Evaluate single pick scenario.
# When picking one container, total openings become:
total_opens_single = total_inhabitants + 1  # 41 + 1 = 42

best_single = None
best_single_profit = -float('inf')
for idx, (multiplier, inh) in enumerate(containers):
    profit = compute_profit(multiplier, inh, total_opens_single)
    if profit > best_single_profit:
        best_single_profit = profit
        best_single = idx

print("Best container if picking ONE (free):")
print(f"Container {best_single} -> Multiplier: {containers[best_single][0]}, "
      f"Inhabitants: {containers[best_single][1]}, Profit: {best_single_profit:.2f}")

# Now evaluate two-container (double pick) scenario.
# When picking two containers, total openings become:
total_opens_double = total_inhabitants + 2  # 41 + 2 = 43

# Cost for opening the second (paid) container:
cost2 = 50000  # <-- Set the SeaShell cost here (adjust as needed)

# We will consider both orders: which container is free and which is paid.
# For any pair (i, j), total profit = profit(i, free) + (profit(j, free) - cost2)
# because in the two-pick scenario both use the same function (with total_opens_double)
# and then we subtract cost2 from one of them.
best_pair = None
best_pair_profit = -float('inf')

n = len(containers)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # If container i is taken as free and container j as paid:
        profit_i = compute_profit(containers[i][0], containers[i][1], total_opens_double)
        profit_j = compute_profit(containers[j][0], containers[j][1], total_opens_double) - cost2
        total_profit = profit_i + profit_j
        if total_profit > best_pair_profit:
            best_pair_profit = total_profit
            best_pair = (i, j)

print("\nBest container pair if picking TWO (first free, second paid):")
print(f"Free pick: Container {best_pair[0]} -> Multiplier: {containers[best_pair[0]][0]}, "
      f"Inhabitants: {containers[best_pair[0]][1]}, Profit: {compute_profit(containers[best_pair[0]][0], containers[best_pair[0]][1], total_opens_double):.2f}")
print(f"Paid pick: Container {best_pair[1]} -> Multiplier: {containers[best_pair[1]][0]}, "
      f"Inhabitants: {containers[best_pair[1]][1]}, Profit (after cost): {compute_profit(containers[best_pair[1]][0], containers[best_pair[1]][1], total_opens_double) - cost2:.2f}")
print(f"Total Profit (pair): {best_pair_profit:.2f}")

# Compare best one-container versus two-container approaches:
if best_single_profit >= best_pair_profit:
    print("\nOptimal Decision: Choose ONE container for maximum profit")
    print(f"-> Pick container {best_single} for a profit of {best_single_profit:.2f}")
else:
    print("\nOptimal Decision: Choose TWO containers for maximum overall profit")
    print(f"-> Pick container {best_pair[0]} for free and container {best_pair[1]} for a cost-adjusted profit, total = {best_pair_profit:.2f}")
