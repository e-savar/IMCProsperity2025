def macaron_strategy(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = "MAGNIFICENT_MACARONS"
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        MAX_HISTORY_LEN = 100
        self.price_history[product].append(mid_price)
        if len(self.price_history[product]) > MAX_HISTORY_LEN:
            self.price_history[product].pop(0)
        position = state.position.get(product, 0)
        position_limit = self.position_limits[product]

        # --- Fair Value Estimation ---
        ewma_alpha = 0.2
        self.ewma_mean[product], self.ewma_var[product] = self.update_ewma(
            mid_price, self.ewma_mean[product], self.ewma_var[product], ewma_alpha
        )
        fair_value = self.ewma_mean[product] or mid_price
        std_dev = max(math.sqrt(self.ewma_var[product] or 1), 1)

        # --- Momentum Signal ---
        momentum_window = 10
        self.last_prices[product].append(mid_price)
        if len(self.last_prices[product]) > momentum_window:
            self.last_prices[product].pop(0)
        momentum = self.last_prices[product][-1] - self.last_prices[product][0] if len(self.last_prices[product]) > 1 else 0

        # --- External Factors: Sunlight & Transport ---
        sunlight = (
            state.observations.plainValueObservations.get("sunlight")
            or getattr(state.observations.conversionObservations.get(product, None), "sunlightIndex", None)
        )
        transport_fees = (
            state.observations.plainValueObservations.get("transport_fees")
            or getattr(state.observations.conversionObservations.get(product, None), "transportFees", None)
        )

        if sunlight is not None:
            self.sunlight_history.append(sunlight)
            self.sunlight_history = self.sunlight_history[-20:]
        if transport_fees is not None:
            self.transport_fees_history.append(transport_fees)
            self.transport_fees_history = self.transport_fees_history[-20:]

        # --- Counterparty Activity Adaptation ---
        recent_trades = state.own_trades.get(product, [])[-5:]  # Get last 5 trades for this product
        recent_counterparties = []

        # Extract counterparty information from recent trades
        for t in recent_trades:
            if t.buyer == "SUBMISSION" and t.seller:
                recent_counterparties.append(t.seller)
            elif t.seller == "SUBMISSION" and t.buyer:
                recent_counterparties.append(t.buyer)

        # Count frequency of each counterparty
        counterparty_freq = {cp: recent_counterparties.count(cp) for cp in set(recent_counterparties)}

        # Adapt strategy based on counterparty activity
        ayumi_sami_active = any(cp in {"Ayumi", "Sami"} for cp in counterparty_freq)
        boris_active = counterparty_freq.get("Boris", 0) >= 2

        # Adjust fair value and position limits based on counterparties
        if ayumi_sami_active:
            fair_value -= 0.5  # Reduce fair value if Ayumi or Sami is active
        if boris_active:
            position_limit += 10  # Increase position limit if Boris is active

        # --- CSI KICKER ---
        CSI = 7.5  # Updated based on better estimate
        sunlight_trend = 0
        if len(self.sunlight_history) >= 5:
            sunlight_trend = self.sunlight_history[-1] - self.sunlight_history[-5]

        sunlight_bias = 0
        sunlight_multiplier = 1.0
        if sunlight is not None:
            sunlight_bias = 0.5 * (CSI - sunlight)
            if sunlight < CSI:
                below_factor = 1 + 0.3 * (CSI - sunlight) ** 2  # Stronger exponential
                if sunlight_trend < 0:
                    below_factor *= (1 + min(-sunlight_trend * 0.4, 0.8))  # Amplify more
                sunlight_multiplier = below_factor
                sunlight_bias *= 2.5  # Stronger base bias
            else:
                sunlight_bias *= 0.4

        fee_bias = 0
        if transport_fees is not None and len(self.transport_fees_history) >= 5:
            avg_fee = sum(self.transport_fees_history[-5:]) / 5
            fee_bias = 0.1 * (transport_fees - avg_fee)

        adjusted_fair_value = fair_value + sunlight_bias + fee_bias

        risk_multiplier = 1.0
        if sunlight is not None:
            if sunlight < CSI:
                risk_multiplier = min(1.8, 1 + 0.25 * (CSI - sunlight))
            else:
                risk_multiplier = max(0.6, 1 - 0.07 * (sunlight - CSI))

        inventory_skew = -0.8 * (position / position_limit) * std_dev
        momentum_skew = 0.6 * (momentum / std_dev) * sunlight_multiplier
        total_skew = inventory_skew + momentum_skew

        base_spread = self.base_spreads[product]
        volatility_factor = 3
        if sunlight is not None and sunlight < CSI:
            volatility_factor *= min(1.8, 1 + 0.15 * (CSI - sunlight))

        spread = base_spread + std_dev * volatility_factor

        bid_price = round(adjusted_fair_value - spread / 2 + total_skew)
        ask_price = round(adjusted_fair_value + spread / 2 + total_skew)

        base_size = self.order_sizes[product]
        adjusted_size = int(base_size * risk_multiplier)

        if sunlight is not None:
            if sunlight < CSI and sunlight_trend < 0:
                buy_size = min(int(adjusted_size * 1.4), position_limit - position)
                sell_size = min(int(adjusted_size * 0.6), position + position_limit)
            elif sunlight > CSI and sunlight_trend > 0:
                buy_size = min(int(adjusted_size * 0.6), position_limit - position)
                sell_size = min(int(adjusted_size * 1.4), position + position_limit)
            else:
                buy_size = min(adjusted_size, position_limit - position)
                sell_size = min(adjusted_size, position + position_limit)
        else:
            buy_size = min(adjusted_size, position_limit - position)
            sell_size = min(adjusted_size, position + position_limit)

        buy_size = int(buy_size)
        sell_size = int(sell_size)

        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))

        return orders
