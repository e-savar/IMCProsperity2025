from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

class Trader:
    def __init__(self):
        # KELP mean reversion parameters
        self.kelp_price_history = []
        self.lookback_window_kelp = 100
        self.min_history_length_kelp = 20
        self.z_threshold_kelp = 1.5
        self.ewma_alpha_kelp = 0.1
        self.ewma_mean_kelp = None
        self.ewma_var_kelp = None
        self.bollinger_width_kelp = 2.0
        self.kelp_last_prices = []
        self.momentum_window_kelp = 5
        self.momentum_weight_kelp = 0.3

        #kelp momentum
        self.weighted_alpha = 0.7  # For weighted moving average
        self.volatility_window = 15  # For volatility calculation
        self.bollinger_width_kelp = 2.0  # For Bollinger Bands


        # RESIN parameters
        self.resin_fair_value = 10000

        # SQUID_INK Z-score strategy parameters
        self.squid_ink_price_history = []
        self.lookback_window_squid_ink = 100
        self.min_history_length_squid_ink = 20
        self.z_threshold_squid_ink = 1.5
        self.ewma_alpha_squid_ink = 0.1
        self.ewma_mean_squid_ink = None
        self.ewma_var_squid_ink = None
        self.bollinger_width_squid_ink = 2.0
        self.squid_ink_last_prices = []
        self.momentum_window_squid_ink = 5
        self.momentum_weight_squid_ink = 0.3

        #kelp momentum
        self.squid_ink_weighted_alpha = 0.7  # For weighted moving average
        self.squid_ink_volatility_window = 15  # For volatility calculation
        self.squid_ink_bollinger_width= 2.0  # For Bollinger Bands


        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}

    def calculate_std_dev(self, data, mean):
        if len(data) <= 1:
            return 0
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)

    def update_ewma(self, price, current_mean, current_var, alpha):
        if current_mean is None:
            return price, 0
        delta = price - current_mean
        new_mean = current_mean + alpha * delta
        new_var = (1 - alpha) * (current_var + alpha * delta * delta)
        return new_mean, new_var
    
    def calculate_wma(self, prices, window, alpha):
        # Ensure there are enough prices for the window size
        if len(prices) < window:
            return np.mean(prices)  # Return the simple mean if not enough data
        
        # Create a weight array that increases from 1 to 'window'
        weights = np.arange(1, window + 1)
        
        # Take the last 'window' number of prices
        recent_prices = prices[-window:]
        
        # Calculate the weighted sum of recent prices
        weighted_sum = np.dot(recent_prices, weights)
        
        # Normalize the weighted sum by the sum of the weights
        wma = weighted_sum / weights.sum()
        
        return wma

    def calculate_volatility(self, prices, window):
        if len(prices) < window:
            return 0
        returns = np.diff(prices[-window:]) / prices[-window:-1]
        return np.std(returns)  # Standard deviation of returns
    
    def update_timeframe_data(self, timestamp, mid_price, product):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp

        if product not in self.timeframe_data:
            self.timeframe_data[product] = {'prices': [], 'volumes': []}

        self.timeframe_data[product]['prices'].append(mid_price)

        if len(self.timeframe_data[product]['prices']) > 500:
            self.timeframe_data[product]['prices'].pop(0)

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int,
                             position_limit: int, product: str, buy_order_volume: int,
                             sell_order_volume: int, fair_value: float, width: int) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    
    def kelp_mean_reversion_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        position = state.position.get("KELP", 0)
        position_limit = 50
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > self.lookback_window_kelp:
            self.kelp_price_history.pop(0)

        self.ewma_mean_kelp, self.ewma_var_kelp = self.update_ewma(
            mid_price, self.ewma_mean_kelp, self.ewma_var_kelp, self.ewma_alpha_kelp
        )
        self.kelp_last_prices.append(mid_price)
        if len(self.kelp_last_prices) > self.momentum_window_kelp:
            self.kelp_last_prices.pop(0)

        if hasattr(state, 'timestamp'):
            self.update_timeframe_data(state.timestamp, mid_price, "KELP")

        if len(self.kelp_price_history) < self.min_history_length_kelp:
            return orders

        simple_mean = sum(self.kelp_price_history) / len(self.kelp_price_history)
        std_dev = self.calculate_std_dev(self.kelp_price_history, simple_mean)
        mean_price = self.ewma_mean_kelp if self.ewma_mean_kelp else simple_mean
        ewma_std_dev = math.sqrt(self.ewma_var_kelp) if self.ewma_var_kelp is not None and self.ewma_var_kelp > 0 else std_dev

        upper_band = mean_price + self.bollinger_width_kelp * ewma_std_dev
        lower_band = mean_price - self.bollinger_width_kelp * ewma_std_dev
        z_score = (mid_price - mean_price) / ewma_std_dev if ewma_std_dev > 0 else 0
        momentum = self.kelp_last_prices[-1] - self.kelp_last_prices[0] if len(self.kelp_last_prices) >= 2 else 0
        adjusted_z = z_score + self.momentum_weight_kelp * (momentum / ewma_std_dev if ewma_std_dev > 0 else 0)

        position_bias = int(position_limit * (math.tanh(adjusted_z / self.z_threshold_kelp)))
        target_position = max(-position_limit, min(position_limit, -position_bias))

        position_diff = target_position - (position + buy_order_volume - sell_order_volume)

        if mid_price < lower_band:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < mean_price:
                    available_volume = -order_depth.sell_orders[ask_price]
                    qty = min(available_volume, position_limit - position - buy_order_volume)
                    if qty > 0:
                        orders.append(Order("KELP", ask_price, qty))
                        buy_order_volume += qty

        if mid_price > upper_band:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > mean_price:
                    available_volume = order_depth.buy_orders[bid_price]
                    qty = min(available_volume, position_limit + position - sell_order_volume)
                    if qty > 0:
                        orders.append(Order("KELP", bid_price, -qty))
                        sell_order_volume += qty

        if position_diff > 0:
            buy_price = best_bid + 1
            qty = min(position_diff, position_limit - position - buy_order_volume)
            if qty > 0:
                orders.append(Order("KELP", buy_price, qty))
        elif position_diff < 0:
            sell_price = best_ask - 1
            qty = min(-position_diff, position_limit + position - sell_order_volume)
            if qty > 0:
                orders.append(Order("KELP", sell_price, -qty))

        return orders
    
    def kelp_momentum_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get("KELP", 0)
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > self.lookback_window_kelp:
            self.kelp_price_history.pop(0)

        # Calculate weighted moving average
        wma_mean = self.calculate_wma(self.kelp_price_history, self.lookback_window_kelp, self.weighted_alpha)

        # Calculate volatility (standard deviation)
        volatility = self.calculate_volatility(self.kelp_price_history, self.volatility_window)

        # Update EWMA mean and variance
        self.ewma_mean_kelp, self.ewma_var_kelp = self.update_ewma(mid_price, self.ewma_mean_kelp, self.ewma_var_kelp, 0.1)

        # Calculate dynamic Bollinger Bands with higher volatility weighting
        std_dev = volatility if volatility > 0 else np.std(self.kelp_price_history)
        upper_band = wma_mean + self.bollinger_width_kelp * 1.5 * std_dev  # Increase the multiplier for wider bands
        lower_band = wma_mean - self.bollinger_width_kelp * 1.5 * std_dev

        # Z-score for mean reversion with increased momentum influence
        z_score = (mid_price - wma_mean) / std_dev if std_dev > 0 else 0
        momentum = self.kelp_last_prices[-1] - self.kelp_last_prices[0] if len(self.kelp_last_prices) >= 2 else 0
        adjusted_z = z_score + (momentum / std_dev if std_dev > 0 else 0)

        # Target position with a higher scaling factor to increase position shifts
        position_bias = int(50 * (adjusted_z / 2))  # Increased the scaling for stronger position bias
        target_position = max(-50, min(50, -position_bias))

        position_diff = target_position - (position + buy_order_volume - sell_order_volume)

        # If the price is too low, buy
        if mid_price < lower_band:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < wma_mean:
                    available_volume = -order_depth.sell_orders[ask_price]
                    qty = min(available_volume, 50 - position - buy_order_volume)
                    qty = min(qty, 50)  # Allow larger position shifts
                    if qty > 0:
                        orders.append(Order("KELP", ask_price, qty))
                        buy_order_volume += qty

        # If the price is too high, sell
        if mid_price > upper_band:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > wma_mean:
                    available_volume = order_depth.buy_orders[bid_price]
                    qty = min(available_volume, 50 + position - sell_order_volume)
                    qty = min(qty, 50)  # Allow larger position shifts
                    if qty > 0:
                        orders.append(Order("KELP", bid_price, -qty))
                        sell_order_volume += qty

        # Adjust order based on volatility (larger orders in higher volatility)
        if position_diff > 0:
            buy_price = best_bid + 1
            qty = min(position_diff, 50 - position - buy_order_volume)
            qty = min(qty, 50)  # Allow larger position shifts
            if qty > 0:
                orders.append(Order("KELP", buy_price, qty))
        elif position_diff < 0:
            sell_price = best_ask - 1
            qty = min(-position_diff, 50 + position - sell_order_volume)
            qty = min(qty, 50)  # Allow larger position shifts
            if qty > 0:
                orders.append(Order("KELP", sell_price, -qty))

        return orders

    # def resin_orders(self, order_depth: OrderDepth, width: int, state: TradingState) -> List[Order]:
    #     orders: List[Order] = []
    #     buy_order_volume = 0
    #     sell_order_volume = 0
    #     position_limit = 50
    #     position = state.position.get("RAINFOREST_RESIN", 0)
    #     fair_value = self.resin_fair_value

    #     try:
    #         baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
    #         bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
    #     except ValueError:
    #         if order_depth.sell_orders:
    #             baaf = min(order_depth.sell_orders.keys())
    #         else:
    #             baaf = fair_value + width

    #         if order_depth.buy_orders:
    #             bbbf = max(order_depth.buy_orders.keys())
    #         else:
    #             bbbf = fair_value - width

    #     if len(order_depth.sell_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -1 * order_depth.sell_orders[best_ask]
    #         if best_ask < fair_value:
    #             quantity = min(best_ask_amount, position_limit - position)
    #             if quantity > 0:
    #                 orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
    #                 buy_order_volume += quantity

    #     if len(order_depth.buy_orders) != 0:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
    #         if best_bid > fair_value:
    #             quantity = min(best_bid_amount, position_limit + position)
    #             if quantity > 0:
    #                 orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
    #                 sell_order_volume += quantity

    #     buy_order_volume, sell_order_volume = self.clear_position_order(
    #         orders, order_depth, position, position_limit,
    #         "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1
    #     )

    #     buy_quantity = position_limit - (position + buy_order_volume)
    #     if buy_quantity > 0:
    #         orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

    #     sell_quantity = position_limit + (position - sell_order_volume)
    #     if sell_quantity > 0:
    #         orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

    #     return orders
    
    def resin_band_strategy(self, order_depth: OrderDepth, width: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        position_limit = 50
        product = "RAINFOREST_RESIN"
        position = state.position.get(product, 0)
        fair_value = self.resin_fair_value
        order_depth = state.order_depths[product]

        # Find best "aggressive" resting opportunities
        try:
            baaf = min([p for p in order_depth.sell_orders if p > fair_value + 1])
            bbbf = max([p for p in order_depth.buy_orders if p < fair_value - 1])
        except ValueError:
            baaf = min(order_depth.sell_orders.keys(), default=fair_value + 3)
            bbbf = max(order_depth.buy_orders.keys(), default=fair_value - 3)

        # --- AGGRESSIVE EXECUTION ON CLEAR MISPRICING ---
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]

            if best_ask < fair_value:
                distance = fair_value - best_ask
                multiplier = 1 + int(distance)  # e.g. 1 at FV-1, 2 at FV-2, etc.
                qty = min(multiplier * best_ask_volume, position_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_order_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]

            if best_bid > fair_value:
                distance = best_bid - fair_value
                multiplier = 1 + int(distance)
                qty = min(multiplier * best_bid_volume, position_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_order_volume += qty

        # --- POSITION MANAGEMENT: flatten strong positions if price allows ---
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit,
            product, buy_order_volume, sell_order_volume, fair_value, 1
        )

        # --- PASSIVE SPREAD CAPTURE (Smart Posting) ---
        spread = baaf - bbbf if baaf and bbbf else 5
        # Only post passives if spread is wide enough to justify
        if spread >= 2:

            buy_qty = position_limit - (position + buy_order_volume)
            if buy_qty > 0:
                price = bbbf + 1  # Closer to mid, increase fill chance
                orders.append(Order(product, price, buy_qty))

            sell_qty = position_limit + (position - sell_order_volume)
            if sell_qty > 0:
                price = baaf - 1
                orders.append(Order(product, price, -sell_qty))

        return orders

    def calculate_moving_average(self, data, window):
        if len(data) < window:
            return None
        return sum(data[-window:]) / window

    def calculate_trend(self, price_history, short_window, long_window):
        if len(price_history) < long_window:
            return 0  # Not enough data to determine trend

        short_ma = self.calculate_moving_average(price_history, short_window)
        long_ma = self.calculate_moving_average(price_history, long_window)

        if short_ma is None or long_ma is None or long_ma == 0:
            return 0

        # Calculate percentage difference
        trend = (short_ma - long_ma) / long_ma
        return trend

    def squidink_orders(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get("SQUID_INK", 0)
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        self.squid_ink_price_history.append(mid_price)
        if len(self.squid_ink_price_history) > self.lookback_window_squid_ink:
            self.squid_ink_price_history.pop(0)

        # Calculate weighted moving average
        squid_ink_wma_mean = self.calculate_wma(self.squid_ink_price_history, self.lookback_window_squid_ink, self.squid_ink_weighted_alpha)

        # Calculate volatility (standard deviation)
        volatility = self.calculate_volatility(self.squid_ink_price_history, self.squid_ink_volatility_window)

        # Update EWMA mean and variance
        self.ewma_mean_squid_ink, self.ewma_var_squid_ink = self.update_ewma(mid_price, self.ewma_mean_squid_ink, self.ewma_var_squid_ink, 0.1)

        # Calculate dynamic Bollinger Bands with higher volatility weighting
        std_dev = volatility if volatility > 0 else np.std(self.squid_ink_price_history)
        upper_band = squid_ink_wma_mean + self.bollinger_width_squid_ink * 1.5 * std_dev  # Increase the multiplier for wider bands
        lower_band = squid_ink_wma_mean - self.bollinger_width_squid_ink * 1.5 * std_dev

        # Z-score for mean reversion with increased momentum influence
        z_score = (mid_price - squid_ink_wma_mean) / std_dev if std_dev > 0 else 0
        momentum = self.squid_ink_last_prices[-1] - self.squid_ink_last_prices[0] if len(self.squid_ink_last_prices) >= 2 else 0
        adjusted_z = z_score + (momentum / std_dev if std_dev > 0 else 0)

        # Target position with a higher scaling factor to increase position shifts
        position_bias = int(50 * (adjusted_z / 2))  # Increased the scaling for stronger position bias
        target_position = max(-50, min(50, -position_bias))

        position_diff = target_position - (position + buy_order_volume - sell_order_volume)

        # If the price is too low, buy
        if mid_price < lower_band:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < squid_ink_wma_mean:
                    available_volume = -order_depth.sell_orders[ask_price]
                    qty = min(available_volume, 50 - position - buy_order_volume)
                    qty = min(qty, 50)  # Allow larger position shifts
                    if qty > 0:
                        orders.append(Order("SQUID_INK", ask_price, qty))
                        buy_order_volume += qty

        # If the price is too high, sell
        if mid_price > upper_band:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > squid_ink_wma_mean:
                    available_volume = order_depth.buy_orders[bid_price]
                    qty = min(available_volume, 50 + position - sell_order_volume)
                    qty = min(qty, 50)  # Allow larger position shifts
                    if qty > 0:
                        orders.append(Order("SQUID_INK", bid_price, -qty))
                        sell_order_volume += qty

        # Adjust order based on volatility (larger orders in higher volatility)
        if position_diff > 0:
            buy_price = best_bid + 1
            qty = min(position_diff, 50 - position - buy_order_volume)
            qty = min(qty, 50)  # Allow larger position shifts
            if qty > 0:
                orders.append(Order("SQUID_INK", buy_price, qty))
        elif position_diff < 0:
            sell_price = best_ask - 1
            qty = min(-position_diff, 50 + position - sell_order_volume)
            qty = min(qty, 50)  # Allow larger position shifts
            if qty > 0:
                orders.append(Order("SQUID_INK", sell_price, -qty))

        return orders


    def run(self, state: TradingState):
        result = {}

        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except Exception:
                trader_data = {}

        if trader_data:
            self.kelp_price_history = trader_data.get("kelp_price_history", self.kelp_price_history)
            self.ewma_mean_kelp = trader_data.get("ewma_mean_kelp", self.ewma_mean_kelp)
            self.ewma_var_kelp = trader_data.get("ewma_var_kelp", self.ewma_var_kelp)
            self.kelp_last_prices = trader_data.get("kelp_last_prices", self.kelp_last_prices)
            self.timeframe_data = trader_data.get("timeframe_data", self.timeframe_data)

        if "RAINFOREST_RESIN" in state.order_depths:
        #     # result["RAINFOREST_RESIN"] = self.resin_orders(state.order_depths["RAINFOREST_RESIN"], 2, state)
            result["RAINFOREST_RESIN"] = self.resin_band_strategy(state.order_depths["RAINFOREST_RESIN"], 2, state)

        if "KELP" in state.order_depths:
            result["KELP"] = self.kelp_mean_reversion_strategy(state.order_depths["KELP"], state)

        if "SQUID_INK" in state.order_depths:
            result["SQUID_INK"] = self.squidink_orders(state.order_depths["SQUID_INK"], state)

        trader_data = {
            "kelp_price_history": self.kelp_price_history,
            "ewma_mean_kelp": self.ewma_mean_kelp,
            "ewma_var_kelp": self.ewma_var_kelp,
            "kelp_last_prices": self.kelp_last_prices,
            "timeframe_data": self.timeframe_data
        }

        return result, 0, jsonpickle.encode(trader_data)