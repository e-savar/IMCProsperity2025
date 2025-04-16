def binomial_tree_call(self, S, K, T, r, sigma, N=5):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    option = [max(price - K, 0) for price in prices]

    for i in reversed(range(N)):
        for j in range(i + 1):
            option[j] = math.exp(-r * dt) * (q * option[j + 1] + (1 - q) * option[j])
    return option[0]

def binomial_tree_greeks(self, S, K, T, r, sigma, N=5):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    option = [max(price - K, 0) for price in prices]

    for i in reversed(range(N - 1, N - 3, -1)):
        for j in range(i + 1):
            option[j] = math.exp(-r * dt) * (q * option[j + 1] + (1 - q) * option[j])

    S_up = S * u
    S_down = S * d
    C_up = option[1]
    C_down = option[0]
    delta = (C_up - C_down) / (S_up - S_down)

    gamma = ((C_up - option[0]) / (S_up - S) - (option[0] - C_down) / (S - S_down)) / ((S_up - S_down) / 2)

    return {"delta": delta, "gamma": gamma}

def volcanic_voucher_strategy(self, state) -> List:
    orders = []

    spot_product = "VOLCANIC_ROCK"
    spot_depth = state.order_depths.get(spot_product, None)
    if not spot_depth or not spot_depth.buy_orders or not spot_depth.sell_orders:
        return orders

    best_bid = max(spot_depth.buy_orders.keys(), default=None)
    best_ask = min(spot_depth.sell_orders.keys(), default=None)
    if best_bid is None or best_ask is None:
        return orders

    spot_price = (best_bid + best_ask) / 2
    T = 5 / 365  # 5 days to expiration
    r = 0.00
    sigma = 0.20 # guess tbh

    vouchers = {
        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750,
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }

    data = []

    for voucher, strike in vouchers.items():
        if voucher not in state.order_depths:
            continue

        depth = state.order_depths[voucher]
        v_bid = max(depth.buy_orders.keys(), default=None)
        v_ask = min(depth.sell_orders.keys(), default=None)
        if v_bid is None or v_ask is None:
            continue

        market_price = (v_bid + v_ask) / 2
        theo_price = self.binomial_tree_call(spot_price, strike, T, r, sigma)
        greeks = self.binomial_tree_greeks(spot_price, strike, T, r, sigma)

        data.append({
            "voucher": voucher,
            "strike": strike,
            "theo_price": theo_price,
            "market_price": market_price,
            "edge": theo_price - market_price,
            "greeks": greeks,
            "bid": v_bid,
            "ask": v_ask,
            "depth": depth
        })

    if not data:
        return orders

    data.sort(key=lambda x: x["edge"])
    cheap = data[-1]
    expensive = data[0]

    position_cheap = state.position.get(cheap["voucher"], 0)
    position_exp = state.position.get(expensive["voucher"], 0)

    buy_qty = min(5, 200 - position_cheap)
    sell_qty = min(5, 200 + position_exp)

    if buy_qty > 0:
        orders.append(Order(cheap["voucher"], cheap["ask"], buy_qty))
    if sell_qty > 0:
        orders.append(Order(expensive["voucher"], expensive["bid"], -sell_qty))

    net_delta = (cheap["greeks"]["delta"] * buy_qty) - (expensive["greeks"]["delta"] * sell_qty)
    hedge_qty = round(-net_delta)
    hedge_limit = 400
    hedge_position = state.position.get(spot_product, 0)

    if hedge_qty > 0 and hedge_position + hedge_qty <= hedge_limit:
        orders.append(Order(spot_product, best_ask, hedge_qty))
    elif hedge_qty < 0 and hedge_position + hedge_qty >= -hedge_limit:
        orders.append(Order(spot_product, best_bid, hedge_qty))

    return orders
