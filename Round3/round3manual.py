import math

# Standard normal CDF approximation (Abramowitz and Stegun formula)
def norm_cdf(x):
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = ((((a5 * k + a4) * k + a3) * k + a2) * k + a1)
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return approx if x >= 0 else 1.0 - approx

# Standard normal PDF
def norm_pdf(x):
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

# Black-Scholes call price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

# Greeks for call options
def black_scholes_greeks(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm_cdf(d1)
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm_pdf(d1) * math.sqrt(T)
    theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T)) 
             - r * K * math.exp(-r * T) * norm_cdf(d2))
    rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }

# Implied Volatility using bisection method
def implied_volatility_call(S, K, T, r, market_price, tol=1e-5, max_iter=100):
    low, high = 0.0001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_call(S, K, T, r, mid)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return mid

def volcanic_voucher_strategy(self, state: TradingState) -> List[Order]:
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
    T = 5 / 365  # 5 days to expiration in year fraction
    r = 0.00

    vouchers = {
        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750,
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }

    iv_data = []

    for voucher, strike in vouchers.items():
        if voucher not in state.order_depths:
            continue

        depth = state.order_depths[voucher]
        v_bid = max(depth.buy_orders.keys(), default=None)
        v_ask = min(depth.sell_orders.keys(), default=None)
        if v_bid is None or v_ask is None:
            continue

        market_price = (v_bid + v_ask) / 2
        iv = implied_volatility_call(spot_price, strike, T, r, market_price)
        greeks = black_scholes_greeks(spot_price, strike, T, r, iv)

        iv_data.append({
            "voucher": voucher,
            "strike": strike,
            "iv": iv,
            "greeks": greeks,
            "price": market_price,
            "bid": v_bid,
            "ask": v_ask,
            "depth": depth
        })

    if not iv_data:
        return orders

    iv_data.sort(key=lambda x: x["iv"])
    cheap = iv_data[0]
    expensive = iv_data[-1]

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