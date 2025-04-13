    def kelp_strat_potential(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get("KELP", 0)
        position_limit = 50
        buy_order_volume = 0
        sell_order_volume = 0
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        
        # Get the best bid and ask prices
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        # Buy logic based on price thresholds
        if mid_price < 9999 and position < position_limit:
            qty = min(20, position_limit - position)  # Buy 20, or however many can be bought
            orders.append(Order("KELP", best_ask, qty))
            buy_order_volume += qty

        if mid_price < 9998 and position < position_limit:
            qty = min(20, position_limit - position - buy_order_volume)  # Buy 20 more if possible
            orders.append(Order("KELP", best_ask, qty))
            buy_order_volume += qty

        if mid_price < 9997 and position < position_limit:
            qty = min(10, position_limit - position - buy_order_volume)  # Buy 10 more if possible
            orders.append(Order("KELP", best_ask, qty))
            buy_order_volume += qty

        # Sell logic based on price thresholds
        if mid_price > 10001 and position > 0:
            qty = min(20, position)  # Sell 20, or however much can be sold
            orders.append(Order("KELP", best_bid, -qty))
            sell_order_volume += qty

        if mid_price > 10002 and position > 0:
            qty = min(20, position - sell_order_volume)  # Sell 20 more if possible
            orders.append(Order("KELP", best_bid, -qty))
            sell_order_volume += qty

        if mid_price > 10003 and position > 0:
            qty = position  # Sell the rest of the position
            orders.append(Order("KELP", best_bid, -qty))
            sell_order_volume += qty

        return orders

    def kelp_market_making_inverted(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Invert best available ask/bid outside fair value range
        try:
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        except ValueError:
            baaf = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fair_value + width
            bbbf = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fair_value - width

        # Inverted: Sell if price is below fair value (expect further decline)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit + position)  # FLIPPED sign
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, -quantity))  # SELL
                    sell_order_volume += quantity

        # Inverted: Buy if price is above fair value (expect further rise)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit - position)  # FLIPPED sign
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, quantity))  # BUY
                    buy_order_volume += quantity

        # Still clear position to reduce risk (you can adjust logic if needed)
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit,
            "KELP", buy_order_volume, sell_order_volume, fair_value, 1
        )

        # Inverted: Passive orders flipped
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, -sell_quantity))  # Place sell where we used to buy

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", baaf - 1, buy_quantity))  # Place buy where we used to sell

        return orders
