from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

class Trader:
    def __init__(self):

        # KELP parameters
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
        self.weighted_alpha = 0.7  # For weighted moving average
        self.volatility_window = 15  # For volatility calculation
        self.bollinger_width_kelp = 2.0  # For Bollinger Bands

        # JAMS parameters
        self.jam_position_limit = 350

        # RAINFOREST_RESIN parameters
        self.resin_fair_value = 10000

        # CROISSANTS parameters
        self.croissant_position_limit = 250
        self.croissant_prices = []
        self.rolling_window = 20
        self.momentum_window = 5
        
        # SQUID_INK strategy parameters
        self.squid_prices = []
        self.squid_window = 10
        self.squid_upper_z = 1.3
        self.squid_lower_z = -1.3
        self.squid_position_limit = 50
        self.squid_ink_weighted_alpha = 0.7  # For weighted moving average
        self.squid_ink_volatility_window = 15  # For volatility calculation
        self.squid_ink_bollinger_width= 2.0  # For Bollinger Bands
        self.squid_volatility_threshold = 0.1  # Volatility threshold for trading decisions

        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}

        # DJEMBES parameters
        self.djembe_position_limit = 60
        self.djembe_price_history = []

        # PICNIC_BASKET_1 parameters
        # contain 6 croissants, 3 jams, 1 DJEMBES
        self.picnic_basket_1_limit = 60

        # PICNIC_BASKET_2 parameters
        # contain 4 croissants, 2 jams
        self.picnic_basket_2_limit = 100

        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}

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
    
    def update_ewma(self, price, current_mean, current_var, alpha):
        if current_mean is None:
            return price, 0
        delta = price - current_mean
        new_mean = current_mean + alpha * delta
        new_var = (1 - alpha) * (current_var + alpha * delta * delta)
        return new_mean, new_var
    
    def update_timeframe_data(self, timestamp, mid_price, product):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp

        if product not in self.timeframe_data:
            self.timeframe_data[product] = {'prices': [], 'volumes': []}

        self.timeframe_data[product]['prices'].append(mid_price)

        if len(self.timeframe_data[product]['prices']) > 500:
            self.timeframe_data[product]['prices'].pop(0)
    
    def calculate_std_dev(self, data, mean):
        if len(data) <= 1:
            return 0
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)
    
    def resin_strategy(self, order_depth: OrderDepth, width: int, state: TradingState) -> List[Order]:
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

    def kelp_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
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
    
    def squidink_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        product = "SQUID_INK"
        position = state.position.get(product, 0)
        position_limit = self.squid_position_limit
        min_order_size = 10  # Minimum order size to avoid placing too small orders

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        if best_bid is None or best_ask is None:
            return orders

        # Calculate VWAP for the product
        total_price = 0
        total_volume = 0
        for price, volume in order_depth.buy_orders.items():
            total_price += price * volume
            total_volume += volume
        for price, volume in order_depth.sell_orders.items():
            total_price += price * volume
            total_volume += volume

        if total_volume == 0:
            return orders

        vwap = total_price / total_volume
        self.squid_prices.append(vwap)

        if len(self.squid_prices) > self.squid_window:
            self.squid_prices.pop(0)

        if len(self.squid_prices) < self.squid_window:
            return orders

        # === Enhanced Strategy Logic ===
        if vwap is None:
            return orders

        # Calculate volatility (standard deviation of recent prices)
        recent_prices = np.array(self.squid_prices[-self.squid_window:])
        volatility = np.std(recent_prices)

        # Calculate order book imbalance
        buy_orders_volume = sum(order_depth.buy_orders.values())
        sell_orders_volume = sum(order_depth.sell_orders.values())
        order_book_imbalance = (buy_orders_volume - sell_orders_volume) / (buy_orders_volume + sell_orders_volume)

        # Use volatility and order book imbalance to adjust trading logic
        if volatility > self.squid_volatility_threshold:
            # High volatility: reduce the order size or refrain from trading
            order_size = min(min_order_size, position_limit - position) if order_book_imbalance > 0 else 0
        else:
            # Low volatility: increase order size and trade more actively
            order_size = min(position_limit - position, 20)

        if best_bid < vwap and order_size > 0:
            orders.append(Order(product, best_ask, -order_size))

        elif best_ask > vwap and order_size > 0:
            orders.append(Order(product, best_bid, order_size)) 

        return orders

    def djembe_strategy(self, order_depth: OrderDepth, state: TradingState):
        symbol = "DJEMBES"
        orders = []
        position = state.position.get(symbol, 0)
        limit = self.djembe_position_limit
        
        # Check for market data
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        
        # Calculate current mid price
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Initialize and update price history
        if not hasattr(self, 'djembe_prices'):
            self.djembe_prices = []
        self.djembe_prices.append(mid_price)
        
        # Wait until we have enough data
        if len(self.djembe_prices) < 300:
            return orders
        
        # Calculate long-term mean
        window_size = 300  # Use last 300 price points
        mean_price = np.mean(self.djembe_prices[-window_size:])
        
        # Calculate standard deviation for determining significant deviations
        std_dev = np.std(self.djembe_prices[-window_size:])
        
        # Calculate z-score (number of standard deviations from mean)
        z_score = (mid_price - mean_price) / std_dev if std_dev > 0 else 0
        
        # Simple position sizing based on deviation
        base_size = int(limit * 0.2)  # 20% of position limit
        
        # Calculate available capacity
        buy_capacity = limit - position
        sell_capacity = limit + position
        
        # Pure mean reversion logic
        if z_score <= -2.0 and buy_capacity > 0:
            # Price significantly below mean - BUY
            buy_price = best_bid + 1  # Aggressive buy
            buy_volume = min(base_size, buy_capacity)
            orders.append(Order(symbol, buy_price, buy_volume))
        
        elif z_score >= 2.0 and sell_capacity > 0:
            # Price significantly above mean - SELL
            sell_price = best_ask - 1  # Aggressive sell
            sell_volume = min(base_size, sell_capacity)
            orders.append(Order(symbol, sell_price, -sell_volume))
        
        # Take profit when price reverts toward mean
        if position > 0 and z_score > 0:
            # We're long and price is now above mean - take profit
            take_profit_volume = min(position, int(position * 0.5))  # Close half position
            orders.append(Order(symbol, best_ask, -take_profit_volume))
        
        elif position < 0 and z_score < 0:
            # We're short and price is now below mean - take profit
            take_profit_volume = min(-position, int(-position * 0.5))  # Close half position
            orders.append(Order(symbol, best_bid, take_profit_volume))
        
        return orders
    def croissants_strategy(self, order_depth: OrderDepth, state: TradingState):
        orders = []
        position = state.position.get("CROISSANTS", 0)
        limit = self.croissant_position_limit

        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None

        if best_ask is None or best_bid is None:
            return orders

        mid_price = (best_ask + best_bid) / 2
        self.croissant_prices.append(mid_price)

        if len(self.croissant_prices) > self.rolling_window:
            self.croissant_prices.pop(0)

        # === Mean + Momentum ===
        rolling_mean = np.mean(self.croissant_prices)
        short_term_momentum = mid_price - self.croissant_prices[-self.momentum_window] if len(self.croissant_prices) >= self.momentum_window else 0
        trending_up = short_term_momentum > 1
        trending_down = short_term_momentum < -1

        # === Strategy Parameters ===
        mean_reversion_edge = 2
        passive_spread = 1
        active_volume = 15
        passive_volume = 5

        # === Active Mean Reversion Entries ===
        # Only act against extremes if there's no strong trend
        if mid_price < rolling_mean - mean_reversion_edge and not trending_down and position < limit:
            buy_price = best_bid + 1
            volume = min(active_volume, limit - position)
            orders.append(Order("CROISSANTS", buy_price, volume))

        if mid_price > rolling_mean + mean_reversion_edge and not trending_up and position > -limit:
            sell_price = best_ask - 1
            volume = min(active_volume, position + limit)
            orders.append(Order("CROISSANTS", sell_price, -volume))

        # === Passive Market Making (Regardless of trend) ===
        if position < limit:
            orders.append(Order("CROISSANTS", int(rolling_mean - passive_spread), passive_volume))
        if position > -limit:
            orders.append(Order("CROISSANTS", int(rolling_mean + passive_spread), -passive_volume))

        return orders

    
        symbol = "DJEMBES"
        orders = []
        position = state.position.get(symbol, 0)
        limit = self.djembe_position_limit
        
        # Check for market data
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        
        # Calculate market metrics
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        spread = best_ask - best_bid
        
        # Calculate order book imbalance
        bid_volume = sum(order_depth.buy_orders.values())
        ask_volume = -sum(order_depth.sell_orders.values())  # Convert to positive
        total_volume = bid_volume + ask_volume
        # Avoid division by zero
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Store historical prices for analysis
        if not hasattr(self, 'djembe_prices'):
            self.djembe_prices = []
        self.djembe_prices.append(mid_price)
        
        # Only proceed when we have enough historical data
        if len(self.djembe_prices) < 20:
            return orders
        
        # Calculate volatility (using standard deviation of returns)
        price_changes = [self.djembe_prices[i] - self.djembe_prices[i-1] for i in range(1, len(self.djembe_prices))]
        volatility = np.std(price_changes[-20:])  # 20-period volatility
        
        # Calculate average daily range (ADR)
        recent_prices = self.djembe_prices[-100:] if len(self.djembe_prices) >= 100 else self.djembe_prices
        adr = max(recent_prices) - min(recent_prices)
        
        # Dynamic parameters based on market conditions
        # Position sizing based on volatility
        base_size = max(1, int(limit * 0.10))  # Base position size (10% of limit)
        vol_adjusted_size = int(base_size * (1 / (1 + volatility)))  # Lower size in higher volatility
        
        # Dynamic price levels based on ADR
        adr_factor = 0.05  # 5% of ADR
        level1 = mid_price - (adr * adr_factor)
        level2 = mid_price - (adr * adr_factor * 2)
        level3 = mid_price + (adr * adr_factor)
        level4 = mid_price + (adr * adr_factor * 2)
        
        # Risk management - reduce position sizes when approaching limits
        risk_factor = min(1.0, (0.8 * limit - abs(position)) / (0.8 * limit))
        risk_factor = max(0.2, risk_factor)  # At least 20% of normal size
        
        # ===== STRATEGY COMPONENTS =====
        
        # 1. ORDER BOOK IMBALANCE STRATEGY
        imbalance_threshold = 0.2
        if imbalance > imbalance_threshold and position < limit * 0.8:
            # Strong buying pressure, go long
            price = best_bid + 1
            size = max(1, int(vol_adjusted_size * risk_factor * min(1.0, imbalance * 2)))
            buy_capacity = limit - position
            if buy_capacity > 0:
                orders.append(Order(symbol, price, min(size, buy_capacity)))
        
        elif imbalance < -imbalance_threshold and position > -limit * 0.8:
            # Strong selling pressure, go short
            price = best_ask - 1
            size = max(1, int(vol_adjusted_size * risk_factor * min(1.0, -imbalance * 2)))
            sell_capacity = limit + position
            if sell_capacity > 0:
                orders.append(Order(symbol, price, -min(size, sell_capacity)))
        
        # 2. MULTI-LEVEL LIQUIDITY PROVISION STRATEGY
        # Calculate remaining capacity
        buy_capacity = limit - position
        sell_capacity = limit + position
        
        # Size scaling by level
        level1_size = max(1, int(vol_adjusted_size * risk_factor * 0.5))
        level2_size = max(1, int(vol_adjusted_size * risk_factor))
        
        # Provide liquidity at multiple levels if we have capacity
        if buy_capacity > 0:
            if int(level1) < best_bid:  # Ensure we don't cross the spread
                orders.append(Order(symbol, int(level1), min(level1_size, buy_capacity)))
                buy_capacity -= level1_size
                
            if buy_capacity > 0 and int(level2) < int(level1):
                orders.append(Order(symbol, int(level2), min(level2_size, buy_capacity)))
        
        if sell_capacity > 0:
            if int(level3) > best_ask:  # Ensure we don't cross the spread
                orders.append(Order(symbol, int(level3), -min(level1_size, sell_capacity)))
                sell_capacity -= level1_size
                
            if sell_capacity > 0 and int(level4) > int(level3):
                orders.append(Order(symbol, int(level4), -min(level2_size, sell_capacity)))
        
        # 3. MEAN REVERSION FOR EXTREME MOVES
        lookback = min(50, len(self.djembe_prices))
        short_ma = np.mean(self.djembe_prices[-10:])
        long_ma = np.mean(self.djembe_prices[-lookback:])
        
        # Only act on significant deviations
        if short_ma < long_ma * 0.95 and position < limit * 0.5:
            # Significantly below MA, buy opportunity
            price = best_bid + 1
            size = max(1, int(vol_adjusted_size * risk_factor * 1.5))
            if buy_capacity > 0:
                orders.append(Order(symbol, price, min(size, buy_capacity)))
                
        elif short_ma > long_ma * 1.05 and position > -limit * 0.5:
            # Significantly above MA, sell opportunity
            price = best_ask - 1
            size = max(1, int(vol_adjusted_size * risk_factor * 1.5))
            if sell_capacity > 0:
                orders.append(Order(symbol, price, -min(size, sell_capacity)))
        
        return orders
    
    def picnic_basket_1_strategy(self, order_depth: OrderDepth, state: TradingState):
        # This strategy should consider the limits and the current order depth
        orders = []

        # get expected price of basket (6x croissants, 3x jams, 1x djembe)
        croissant_best_bid = max(state.order_depths["CROISSANTS"].buy_orders.keys())
        croissant_best_ask = min(state.order_depths["CROISSANTS"].sell_orders.keys())
        croissant_price = (croissant_best_ask + croissant_best_bid) / 2

        jam_best_bid = max(state.order_depths["JAMS"].buy_orders.keys())
        jam_best_ask = min(state.order_depths["JAMS"].sell_orders.keys())
        jam_price = (jam_best_ask + jam_best_bid) / 2

        djembe_best_bid = max(state.order_depths["DJEMBES"].buy_orders.keys())
        djembe_best_ask = min(state.order_depths["DJEMBES"].sell_orders.keys())
        djembe_price = (djembe_best_ask + djembe_best_bid) / 2

        # calculate expected price of basket
        expected_basket_price = (6 * croissant_price) + (3 * jam_price) + djembe_price

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]

        threshold = 0.001
        if best_ask < expected_basket_price * (1 - threshold):
            # buy basket
            quantity = min(self.picnic_basket_1_limit, best_ask_volume)
            orders.append(Order("PICNIC_BASKET1", best_ask, quantity))
        elif best_bid > expected_basket_price * (1 + threshold):
            # sell basket
            quantity = min(self.picnic_basket_1_limit, best_bid_volume)
            orders.append(Order("PICNIC_BASKET1", best_bid, -quantity))

        return orders

    def picnic_basket_2_strategy(self, order_depth: OrderDepth, state: TradingState):
        # This strategy should consider the limits and the current order depth
        orders = []

        # get expected price of basket (4x croissants, 2x jams)
        croissant_best_bid = max(state.order_depths["CROISSANTS"].buy_orders.keys())
        croissant_best_ask = min(state.order_depths["CROISSANTS"].sell_orders.keys())
        croissant_price = (croissant_best_ask + croissant_best_bid) / 2

        jam_best_bid = max(state.order_depths["JAMS"].buy_orders.keys())
        jam_best_ask = min(state.order_depths["JAMS"].sell_orders.keys())
        jam_price = (jam_best_ask + jam_best_bid) / 2

        # calculate expected price of basket
        expected_basket_price = (4 * croissant_price) + (2 * jam_price)

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]

        threshold = 0.001
        if best_ask < expected_basket_price * (1 - threshold):
            # buy basket
            quantity = min(self.picnic_basket_2_limit, best_ask_volume)
            orders.append(Order("PICNIC_BASKET2", best_ask, quantity))
        elif best_bid > expected_basket_price * (1 + threshold):
            # sell basket
            quantity = min(self.picnic_basket_2_limit, best_bid_volume)
            orders.append(Order("PICNIC_BASKET2", best_bid, -quantity))

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

        # if "RAINFOREST_RESIN" in state.order_depths:
        #     result["RAINFOREST_RESIN"] = self.resin_strategy(state.order_depths["RAINFOREST_RESIN"], 2, state)

        # if "KELP" in state.order_depths:
        #     result["KELP"] = self.kelp_strategy(state.order_depths["KELP"], state)
        
        # if "SQUID_INK" in state.order_depths:
        #     result["SQUID_INK"] = self.squidink_strategy(state.order_depths["SQUID_INK"], state)
        
        # if "JAMS" in state.order_depths:
        #     result["JAMS"] = self.jams_strategy(state.order_depths["JAMS"], state)

        # if "CROISSANTS" in state.order_depths:
        #     result["CROISSANTS"] = self.croissants_strategy(state.order_depths["CROISSANTS"], state)

        # if "DJEMBES" in state.order_depths:
        #     result["DJEMBES"] = self.djembe_strategy(state.order_depths["DJEMBES"], state)

        # if "PICNIC_BASKET1" in state.order_depths:
        #     result["PICNIC_BASKET1"] = self.picnic_basket_1_strategy(state.order_depths["PICNIC_BASKET1"], state)

        # if "PICNIC_BASKET2" in state.order_depths:
        #     result["PICNIC_BASKET2"] = self.picnic_basket_2_strategy(state.order_depths["PICNIC_BASKET2"], state)
        trader_data = {
            "kelp_price_history": self.kelp_price_history,
            "ewma_mean_kelp": self.ewma_mean_kelp,
            "ewma_var_kelp": self.ewma_var_kelp,
            "kelp_last_prices": self.kelp_last_prices,
            "timeframe_data": self.timeframe_data
        }

        return result, 0, jsonpickle.encode(trader_data)
    