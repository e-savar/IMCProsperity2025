from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

# Needed for visualizer and backtester provided by Jmerle
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):

        # RAINFOREST_RESIN parameters
        self.resin_fair_value = 10000

        # CROISSANTS parameters
        self.croissant_position_limit = 250
        self.croissant_prices = []
        self.rolling_window = 20
        self.momentum_window = 5
        
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

        # commodity parameters
        self.position_limits = {
            "KELP": 50,
            "SQUID_INK": 50,
            "JAMS": 350,
            'CROISSANTS': 250,
        }
        self.base_spreads = {
            "KELP": 1.3,
            "SQUID_INK": 1.7,
            "JAMS": 1.5,
            'CROISSANTS': 1.5,
        }
        self.inventory_factor = {
            "KELP": -0.2,
            "SQUID_INK": -0.1,
            "JAMS": -0.25,
            'CROISSANTS': -0.4,
        }
        self.momentum_factor = {
            "KELP": 0.1,
            "SQUID_INK": 0.2,
            "JAMS": 0.5,
            'CROISSANTS': 0.25,
        }
        self.ewma_mean = {
            "KELP": None,
            "SQUID_INK": None,
            "JAMS": None,
            "CROISSANTS": None,
        }
        self.ewma_var = {
            "KELP": None,
            "SQUID_INK": None,
            "JAMS": None,
            "CROISSANTS": None,
        }
        self.price_history = {
            "KELP": [],
            "SQUID_INK": [],
            "JAMS": [],
            "CROISSANTS": [],
        }
        self.last_prices = {
            "KELP": [],
            "SQUID_INK": [],
            "JAMS": [],
            "CROISSANTS": [],
        }
        self.volatility_spread_multiplier = 1.7
        self.lookback_window = 100
        self.ewma_alpha = 0.2


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

    def commodity_strategy(self, product: str, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        position = state.position.get(product, 0)
        position_limit = self.position_limits[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.price_history[product].append(mid_price)

        if len(self.price_history[product]) > self.lookback_window:
            self.price_history[product].pop(0)
        self.ewma_mean[product], self.ewma_var[product] = self.update_ewma(
            mid_price, self.ewma_mean[product], self.ewma_var[product], self.ewma_alpha
        )

        fair_value = self.ewma_mean[product] if self.ewma_mean[product] else mid_price
        std_dev = math.sqrt(self.ewma_var[product]) if self.ewma_var[product] and self.ewma_var[product] > 0 else 1

        inventory_skew = self.inventory_factor[product] * position / position_limit
        self.last_prices[product].append(mid_price)
        if len(self.last_prices[product]) > self.momentum_window:
            self.last_prices[product].pop(0)
        momentum = self.last_prices[product][-1] - self.last_prices[product][0] if len(self.last_prices[product]) >= 2 else 0
        momentum_skew = self.momentum_factor[product] * (momentum / std_dev)
        total_skew = inventory_skew + momentum_skew

        base_spread = self.base_spreads[product]
        dynamic_spread = base_spread + std_dev * self.volatility_spread_multiplier

        quote_bid = int(fair_value - dynamic_spread / 2 + total_skew)
        quote_ask = int(fair_value + dynamic_spread / 2 + total_skew)
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
            self.timeframe_data = trader_data.get("timeframe_data", self.timeframe_data)
            self.djembe_theoretical_prices = trader_data.get("djembe_theoretical_prices", self.djembe_theoretical_prices)
            self.djembe_actual_prices = trader_data.get("djembe_actual_prices", self.djembe_actual_prices)
            self.ewma_mean = trader_data.get("ewma_mean", self.ewma_mean)
            self.ewma_var = trader_data.get("ewma_var", self.ewma_var)
            self.price_history = trader_data.get("price_history", self.price_history)
            self.last_prices = trader_data.get("last_prices", self.last_prices)

        # if "RAINFOREST_RESIN" in state.order_depths:
        #     result["RAINFOREST_RESIN"] = self.resin_strategy(state.order_depths["RAINFOREST_RESIN"], 2, state)
        
        # if "KELP" in state.order_depths:
        #     result["KELP"] = self.commodity_strategy("KELP", state.order_depths["KELP"], state)
                
        # if "SQUID_INK" in state.order_depths:
        #     result["SQUID_INK"] = self.commodity_strategy("SQUID_INK", state.order_depths["SQUID_INK"], state)

        # # OPTIMIZE parameters        
        # if "JAMS" in state.order_depths:
        #     result["JAMS"] = self.commodity_strategy("JAMS", state.order_depths["JAMS"], state)
        
        if "CROISSANTS" in state.order_depths:
            result["CROISSANTS"] = self.commodity_strategy("CROISSANTS", state.order_depths["CROISSANTS"], state)
        
        # if "DJEMBES" in state.order_depths: 
        #     result["DJEMBES"] = self.djembes_strategy(state.order_depths["DJEMBES"], state)
        
        # if "PICNIC_BASKET1" in state.order_depths:
        #     result["PICNIC_BASKET1"] = self.picnic_basket_1_strategy(state.order_depths["PICNIC_BASKET1"], state)
        
        # if "PICNIC_BASKET2" in state.order_depths:
        #     result["PICNIC_BASKET2"] = self.picnic_basket_2_strategy(state.order_depths["PICNIC_BASKET2"], state)
                
        trader_data = {
            "timeframe_data": self.timeframe_data,
            "djembe_theoretical_prices": self.djembe_theoretical_prices,
            "djembe_actual_prices": self.djembe_actual_prices,
            "ewma_mean": self.ewma_mean,
            "ewma_var": self.ewma_var,
            "price_history": self.price_history,
            "last_prices": self.last_prices,

        }
        
        logger.flush(state, result, conversions, jsonpickle.encode(trader_data))
        # Return encoded trader data for state persistence
        return result, conversions, jsonpickle.encode(trader_data)
    