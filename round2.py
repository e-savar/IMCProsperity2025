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
        self.z_threshold_kelp = 1.3
        self.ewma_alpha_kelp = 0.2
        self.ewma_mean_kelp = None
        self.ewma_var_kelp = None
        self.bollinger_width_kelp = 2
        self.kelp_last_prices = []
        self.momentum_window_kelp = 5
        self.momentum_weight_kelp = 0.3
        self.weighted_alpha = 0.7  # For weighted moving average
        self.volatility_window = 15  # For volatility calculation
        self.volatility_spread_multiplier_kelp = 1.7

        # JAMS parameters
        self.jams_position_limit = 350
        self.jams_prices = []
        self.jams_window = 10
        self.jams_upper_z = 1.3
        self.jams_lower_z = -1.3 

        # RAINFOREST_RESIN parameters
        self.resin_fair_value = 10000

        # CROISSANTS parameters
        self.croissant_position_limit = 250
        self.croissant_prices = []
        self.rolling_window = 20
        self.momentum_window = 5
        
        # SQUID_INK strategy parameters
        self.squid_price_history = []
        self.lookback_window_squid = 100
        self.min_history_length_squid = 20
        self.z_threshold_squid = 1.3
        self.ewma_alpha_squid = 0.2
        self.ewma_mean_squid = None
        self.ewma_var_squid = None
        self.bollinger_width_squid = 2
        self.squid_last_prices = []
        self.momentum_window_squid = 5
        self.momentum_weight_squid = 0.3
        self.weighted_alpha = 0.7  # For weighted moving average
        self.volatility_window = 15  # For volatility calculation
        self.volatility_spread_multiplier_squid = 1.7
        self.squid_position_limit = 50

        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}

        # DJEMBES parameters
        self.djembe_position_limit = 60
        self.djembe_theoretical_prices = []
        self.djembe_actual_prices = []
        self.djembe_window = 20  # Lookback window size
        self.djembe_spread_history = []

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
    
    # permanent
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

    # permanent
    def kelp_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        position = state.position.get("KELP", 0)
        position_limit = 50

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > self.lookback_window_kelp:
            self.kelp_price_history.pop(0)

        self.ewma_mean_kelp, self.ewma_var_kelp = self.update_ewma(
            mid_price, self.ewma_mean_kelp, self.ewma_var_kelp, self.ewma_alpha_kelp
        )
        fair_value = self.ewma_mean_kelp if self.ewma_mean_kelp else mid_price
        std_dev = math.sqrt(self.ewma_var_kelp) if self.ewma_var_kelp and self.ewma_var_kelp > 0 else 1

        # INVENTORY-BASED SKEW
        inventory_skew = -0.2 * position / position_limit  # skew is stronger the closer to position_limit

        # MOMENTUM SKEW (optional)
        self.kelp_last_prices.append(mid_price)
        if len(self.kelp_last_prices) > self.momentum_window_kelp:
            self.kelp_last_prices.pop(0)
        momentum = self.kelp_last_prices[-1] - self.kelp_last_prices[0] if len(self.kelp_last_prices) >= 2 else 0
        momentum_skew = 0.1 * (momentum / std_dev)

        total_skew = inventory_skew + momentum_skew

        # DYNAMIC SPREAD (wider when volatile)
        base_spread = 1.3
        dynamic_spread = base_spread + std_dev * self.volatility_spread_multiplier_kelp

        quote_bid = int(fair_value - dynamic_spread / 2 + total_skew)
        quote_ask = int(fair_value + dynamic_spread / 2 + total_skew)

        # Determine order sizes (less aggressive when near limits)
        bid_volume = min(10, position_limit - position)
        ask_volume = min(10, position_limit + position)

        if bid_volume > 0:
            orders.append(Order("KELP", quote_bid, bid_volume))
        if ask_volume > 0:
            orders.append(Order("KELP", quote_ask, -ask_volume))

        return orders
   
    # stable - try working on round 2 day 1
    def squidink_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = "SQUID_INK"
        position = state.position.get(product, 0)
        position_limit = self.squid_position_limit

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        # Price history for moving stats
        self.squid_price_history.append(mid_price)
        if len(self.squid_price_history) > self.lookback_window_squid:
            self.squid_price_history.pop(0)

        # EWMA mean/var update
        self.ewma_mean_squid, self.ewma_var_squid = self.update_ewma(
            mid_price, self.ewma_mean_squid, self.ewma_var_squid, self.ewma_alpha_squid
        )
        fair_value = self.ewma_mean_squid if self.ewma_mean_squid else mid_price
        std_dev = math.sqrt(self.ewma_var_squid) if self.ewma_var_squid and self.ewma_var_squid > 0 else 1

        # Inventory-based skew
        inventory_skew = -0.1 * position / position_limit

        # Momentum skew
        self.squid_last_prices.append(mid_price)
        if len(self.squid_last_prices) > self.momentum_window_squid:
            self.squid_last_prices.pop(0)
        momentum = self.squid_last_prices[-1] - self.squid_last_prices[0] if len(self.squid_last_prices) >= 2 else 0
        momentum_skew = 0.2 * (momentum / std_dev)

        total_skew = inventory_skew + momentum_skew

        # Dynamic spread
        base_spread = 1.7
        dynamic_spread = base_spread + std_dev * self.volatility_spread_multiplier_squid

        quote_bid = int(fair_value - dynamic_spread / 2 + total_skew)
        quote_ask = int(fair_value + dynamic_spread / 2 + total_skew)

        # Order sizing
        bid_volume = min(10, position_limit - position)
        ask_volume = min(10, position_limit + position)

        if bid_volume > 0:
            orders.append(Order(product, quote_bid, bid_volume))
        if ask_volume > 0:
            orders.append(Order(product, quote_ask, -ask_volume))

        return orders

    def djembes_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        product = "DJEMBES"  # Corrected product name
        basket1 = "PICNIC_BASKET1"
        basket2 = "PICNIC_BASKET2"
        position = state.position.get(product, 0)
        position_limit = 60  # Specified position limit
        
        # Check if we have all required products in order depths
        if basket1 not in state.order_depths or basket2 not in state.order_depths:
            return orders
        
        # Get depths for baskets
        b1_depth = state.order_depths[basket1]
        b2_depth = state.order_depths[basket2]
        
        # Check if we have both bids and asks for both baskets
        if (not b1_depth.buy_orders or not b1_depth.sell_orders or
            not b2_depth.buy_orders or not b2_depth.sell_orders or
            not order_depth.buy_orders or not order_depth.sell_orders):
            return orders
        
        # Get mid prices
        b1_best_bid = max(b1_depth.buy_orders.keys())
        b1_best_ask = min(b1_depth.sell_orders.keys())
        b1_mid = (b1_best_bid + b1_best_ask) / 2
        
        b2_best_bid = max(b2_depth.buy_orders.keys())
        b2_best_ask = min(b2_depth.sell_orders.keys())
        b2_mid = (b2_best_bid + b2_best_ask) / 2
        
        dj_best_bid = max(order_depth.buy_orders.keys())
        dj_best_ask = min(order_depth.sell_orders.keys())
        dj_mid = (dj_best_bid + dj_best_ask) / 2
        
        # Calculate theoretical DJEMBES price using the formula
        # 2 * Price(Basket1) - 3 * Price(Basket2) = 2 * Price(DJEMBES)
        # Therefore: DJEMBES_theoretical = (2 * Price(Basket1) - 3 * Price(Basket2)) / 2
        dj_theoretical = (2 * b1_mid - 3 * b2_mid) / 2
        
        # Add theoretical price to history
        if not hasattr(self, 'djembes_theoretical_prices'):
            self.djembes_theoretical_prices = []
            self.djembes_actual_prices = []
            self.djembes_window = 20  # Default window size
            self.djembe_spread_history = []  # Initialize the spread history
        
        self.djembes_theoretical_prices.append(dj_theoretical)
        self.djembes_actual_prices.append(dj_mid)
        
        # Keep only the recent window of prices
        if len(self.djembes_theoretical_prices) > self.djembes_window:
            self.djembes_theoretical_prices.pop(0)
            self.djembes_actual_prices.pop(0)
            self.djembe_spread_history.pop(0)
        
        # Price difference between theoretical and actual
        price_diff = dj_theoretical - dj_mid
        self.djembe_spread_history.append(price_diff)
        
        # Simple threshold-based trading with fixed thresholds
        buy_threshold = 1.5
        sell_threshold = 1.5
        
        # Base order size
        order_size = 1
        
        # Trading logic
        if price_diff > buy_threshold:
            # DJEMBES is undervalued based on basket prices
            if position < position_limit:
                buy_volume = min(order_size, position_limit - position)
                # Place a buy order at the best bid or slightly higher
                buy_price = dj_best_bid + 1  # Try matching the current best bid first
                orders.append(Order(product, buy_price, buy_volume))
        
        if price_diff < -sell_threshold:
            # DJEMBES is overvalued based on basket prices
            if position > -position_limit:
                sell_volume = min(order_size, position_limit + position)  # Ensure non-negative volume
                if sell_volume > 0:
                    sell_price = dj_best_ask - 1
                    orders.append(Order(product, sell_price, -sell_volume))

        # Ensure that the spread history has enough data before calculating z-score
        if len(self.djembe_spread_history) > 5:
            spread_std = np.std(self.djembe_spread_history) + 1e-9  # Avoid division by zero
            spread_mean = np.mean(self.djembe_spread_history)
            z_score = (price_diff - spread_mean) / spread_std
            
            # Check for arbitrage opportunity
            threshold = 1.3
            if z_score > threshold:
                # Arbitrage opportunity detected, buy (undervalued)
                if position < position_limit:
                    buy_volume = min(order_size, position_limit - position)
                    buy_price = dj_best_bid
                    orders.append(Order(product, buy_price, buy_volume))
            elif z_score < -threshold:
                # Arbitrage opportunity detected, sell (overvalued)
                if position > -position_limit:
                    sell_volume = min(order_size, position_limit + position)
                    sell_price = dj_best_ask
                    orders.append(Order(product, sell_price, -sell_volume))
        
        # Add simple market making if no arbitrage opportunity detected
        if not orders and len(self.djembes_theoretical_prices) > 5:
            spread = 2  # Tight spread for market making
            
            if position < position_limit:
                bid_price = int(dj_theoretical - spread)
                # Make sure our bid is not above the best ask
                if bid_price >= dj_best_ask:
                    bid_price = dj_best_ask - 1
                
                buy_volume = min(5, position_limit - position)
                if bid_price > 0:  # Ensure we're not bidding at negative prices
                    orders.append(Order(product, bid_price, buy_volume))
            
            if position > -position_limit:
                ask_price = int(dj_theoretical + spread)
                # Make sure our ask is not below the best bid
                if ask_price <= dj_best_bid:
                    ask_price = dj_best_bid + 1
                    
                sell_volume = min(5, position_limit + position)
                orders.append(Order(product, ask_price, -sell_volume))
        
        return orders

    def jams_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
            orders = []
            product = "JAMS"
            position = state.position.get(product, 0)
            position_limit = self.jams_position_limit

            if not order_depth.buy_orders or not order_depth.sell_orders:
                return orders

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            if best_bid is None or best_ask is None:
                return orders

            mid_price = (best_bid + best_ask) / 2
            self.jams_prices.append(mid_price)

            if len(self.jams_prices) > self.jams_window:
                self.jams_prices.pop(0)

            if len(self.jams_prices) < self.jams_window:
                return orders

            fair_price = np.mean(self.jams_prices[-self.jams_window:])
            price_array = np.array(self.jams_prices[-self.jams_window:])
            time_array = np.arange(len(price_array))
            slope, _ = np.polyfit(time_array, price_array, 1)

            momentum_bias_factor = 1.5 #conservative?
            fair_price += max(0, momentum_bias_factor * slope)  

            # limit volatility
            volatility = np.std(price_array)
            max_volatility = 3 
            if volatility > max_volatility:
                return orders  

            price_deviations = np.abs(price_array - fair_price)
            mean_deviation = np.mean(price_deviations)
            std_deviation = np.std(price_deviations)

            z_score = 0
            if std_deviation > 0:
                current_deviation = abs(mid_price - fair_price)
                z_score = (current_deviation - mean_deviation) / std_deviation

            base_spread = 16  # too wider?
            spread_sensitivity = 8
            spread = base_spread + spread_sensitivity * z_score

            # size scaling
            max_order_size = 50
            if slope < 0:
                order_size = int(max_order_size * (1 + slope))  
                order_size = max(order_size, 10) 
            else:
                order_size = max_order_size

            bid_price = int(fair_price - spread)
            ask_price = int(fair_price + spread)

            # do not buy avoid buying in strong downtrend
            if slope >= -0.3 and position < position_limit:
                buy_volume = min(order_size, position_limit - position)
                orders.append(Order(product, bid_price, buy_volume))

            if position > -position_limit:
                sell_volume = min(order_size, position + position_limit)
                orders.append(Order(product, ask_price, -sell_volume))

            return orders
    
    def croissants_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        symbol = "CROISSANTS"
        position = state.position.get(symbol, 0)
        limit = self.croissant_position_limit

        # Ensure there are valid buy and sell orders
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        # Get the best bid and ask prices
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        if best_bid is None or best_ask is None:
            return orders

        # Calculate the mid-price (fair value estimate)
        mid_price = (best_bid + best_ask) / 2

        # Track mid prices for moving average calculation
        self.croissant_prices.append(mid_price)

        # Keep only the last 'rolling_window' prices
        if len(self.croissant_prices) > self.rolling_window:
            self.croissant_prices.pop(0)

        # If there aren't enough prices, do nothing
        if len(self.croissant_prices) < self.rolling_window:
            return orders

        # Calculate moving average of the mid prices
        mean_price = np.mean(self.croissant_prices)

        # === Dynamic Spread Calculation Based on Price Deviations ===
        price_deviations = np.abs(np.array(self.croissant_prices) - mean_price)
        mean_deviation = np.mean(price_deviations)
        std_deviation = np.std(price_deviations)

        # If no variation, use a fixed spread
        if std_deviation == 0:
            spread = 2
        else:
            # Use mean deviation to adjust spread more conservatively
            spread = max(2, mean_deviation)  # Use mean deviation for more conservative spread adjustment

        # === Market Making Logic ===

        # BUY SIDE
        if position < limit:
            buy_price = int(mean_price - spread)
            buy_volume = min(5, limit - position)  # 5 is the fixed order size
            if buy_volume > 0:
                orders.append(Order(symbol, buy_price, buy_volume))

        # SELL SIDE
        if position > -limit:
            sell_price = int(mean_price + spread)
            sell_volume = min(5, position + limit)  # 5 is the fixed order size
            if sell_volume > 0:
                orders.append(Order(symbol, sell_price, -sell_volume))

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

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
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
            self.djembe_theoretical_prices = trader_data.get("djembe_theoretical_prices", self.djembe_theoretical_prices)
            self.djembe_actual_prices = trader_data.get("djembe_actual_prices", self.djembe_actual_prices)
            self.squid_price_history = trader_data.get("squid_price_history", self.squid_price_history)
            self.ewma_mean_squid = trader_data.get("ewma_mean_squid", self.ewma_mean_squid)
            self.ewma_var_squid = trader_data.get("ewma_var_squid", self.ewma_var_squid)
            self.squid_last_prices = trader_data.get("squid_last_prices", self.squid_last_prices)
        
        # if "RAINFOREST_RESIN" in state.order_depths:
        #     result["RAINFOREST_RESIN"] = self.resin_strategy(state.order_depths["RAINFOREST_RESIN"], 2, state)
        
        # if "KELP" in state.order_depths:
        #     result["KELP"] = self.kelp_strategy(state.order_depths["KELP"], state)
                
        if "SQUID_INK" in state.order_depths:
            result["SQUID_INK"] = self.squidink_strategy(state.order_depths["SQUID_INK"], state)
                
        # if "JAMS" in state.order_depths:
        #     result["JAMS"] = self.jams_strategy(state.order_depths["JAMS"], state)
        
        # if "CROISSANTS" in state.order_depths:
        #     result["CROISSANTS"] = self.croissants_strategy(state.order_depths["CROISSANTS"], state)
        
        # if "DJEMBES" in state.order_depths: 
        #     result["DJEMBES"] = self.djembes_strategy(state.order_depths["DJEMBES"], state)
        
        # if "PICNIC_BASKET1" in state.order_depths:
        #     result["PICNIC_BASKET1"] = self.picnic_basket_1_strategy(state.order_depths["PICNIC_BASKET1"], state)
        
        # if "PICNIC_BASKET2" in state.order_depths:
        #     result["PICNIC_BASKET2"] = self.picnic_basket_2_strategy(state.order_depths["PICNIC_BASKET2"], state)
                
        trader_data = {
            "kelp_price_history": self.kelp_price_history,
            "ewma_mean_kelp": self.ewma_mean_kelp,
            "ewma_var_kelp": self.ewma_var_kelp,
            "kelp_last_prices": self.kelp_last_prices,
            "timeframe_data": self.timeframe_data,
            "djembe_theoretical_prices": self.djembe_theoretical_prices,
            "djembe_actual_prices": self.djembe_actual_prices,
            "squid_price_history": self.squid_price_history,
            "ewma_mean_squid": self.ewma_mean_squid,
            "ewma_var_squid": self.ewma_var_squid,
            "squid_last_prices": self.squid_last_prices,
        }
        
        return result, conversions, jsonpickle.encode(trader_data)
    