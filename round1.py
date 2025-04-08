from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        # KELP mean reversion parameters
        self.kelp_price_history = []
        self.lookback_window = 100  # Store last 100 timestamps (adjust based on your data)
        self.min_history_length = 20  # Minimum data points before trading
        self.z_threshold = 1.5  # Signal threshold (how many std deviations)
        self.ewma_alpha = 0.1  # Weight for exponential moving average
        self.ewma_mean = None  # Exponential weighted moving average
        self.ewma_var = None   # Exponential weighted variance
        self.bollinger_width = 2.0  # Width of Bollinger Bands
        self.kelp_last_prices = []  # Track recent prices for momentum
        self.momentum_window = 5    # Window for momentum calculation
        
        # RESIN parameters (keeping basic implementation)
        self.resin_fair_value = 10000
        
        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}

    def calculate_std_dev(self, data, mean):
        if len(data) <= 1:
            return 0
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)

    def update_ewma(self, price):
        if self.ewma_mean is None:
            self.ewma_mean = price
            self.ewma_var = 0
            return
        
        # EWMA mean
        delta = price - self.ewma_mean
        self.ewma_mean += self.ewma_alpha * delta
        
        # EWMA variance
        self.ewma_var = (1 - self.ewma_alpha) * (self.ewma_var + self.ewma_alpha * delta * delta)
    
    def update_timeframe_data(self, timestamp, mid_price, product):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
        
        if product not in self.timeframe_data:
            self.timeframe_data[product] = {'prices': [], 'volumes': []}
        
        self.timeframe_data[product]['prices'].append(mid_price)
        
        # only recent data
        if len(self.timeframe_data[product]['prices']) > 500:
            self.timeframe_data[product]['prices'].pop(0)

    def kelp_mean_reversion_strategy(self, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        position = state.position.get("KELP", 0)
        position_limit = 20
        buy_order_volume = 0
        sell_order_volume = 0

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > self.lookback_window:
            self.kelp_price_history.pop(0)

        self.update_ewma(mid_price)
        self.kelp_last_prices.append(mid_price)
        if len(self.kelp_last_prices) > self.momentum_window:
            self.kelp_last_prices.pop(0)

        if hasattr(state, 'timestamp'):
            self.update_timeframe_data(state.timestamp, mid_price, "KELP")

        if len(self.kelp_price_history) < self.min_history_length:
            return orders

        simple_mean = sum(self.kelp_price_history) / len(self.kelp_price_history)
        std_dev = self.calculate_std_dev(self.kelp_price_history, simple_mean)
        mean_price = self.ewma_mean if self.ewma_mean else simple_mean
        ewma_std_dev = math.sqrt(self.ewma_var) if self.ewma_var > 0 else std_dev

        upper_band = mean_price + self.bollinger_width * ewma_std_dev
        lower_band = mean_price - self.bollinger_width * ewma_std_dev
        z_score = (mid_price - mean_price) / ewma_std_dev if ewma_std_dev > 0 else 0
        momentum = self.kelp_last_prices[-1] - self.kelp_last_prices[0] if len(self.kelp_last_prices) >= 3 else 0
        adjusted_z = z_score + 0.3 * (momentum / ewma_std_dev if ewma_std_dev > 0 else 0)

        # Dynamic position sizing using scaled sigmoid
        position_bias = int(position_limit * (math.tanh(adjusted_z / self.z_threshold)))
        target_position = max(-position_limit, min(position_limit, -position_bias))

        position_diff = target_position - (position + buy_order_volume - sell_order_volume)

        # Take advantage of mispriced asks
        if mid_price < lower_band:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < mean_price:
                    available_volume = -order_depth.sell_orders[ask_price]
                    qty = min(available_volume, position_limit - position - buy_order_volume)
                    if qty > 0:
                        orders.append(Order("KELP", ask_price, qty))
                        buy_order_volume += qty

        # Take advantage of mispriced bids
        if mid_price > upper_band:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > mean_price:
                    available_volume = order_depth.buy_orders[bid_price]
                    qty = min(available_volume, position_limit + position - sell_order_volume)
                    if qty > 0:
                        orders.append(Order("KELP", bid_price, -qty))
                        sell_order_volume += qty

        # Execution for any remaining imbalance toward target
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

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        # Keeping your original resin implementation
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> tuple:
        # Keeping your original implementation
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
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

    def run(self, state: TradingState):
        result = {}

        # Initialize trader data if available
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
                self.kelp_price_history = trader_data.get("kelp_price_history", self.kelp_price_history)
                self.ewma_mean = trader_data.get("ewma_mean", self.ewma_mean)
                self.ewma_var = trader_data.get("ewma_var", self.ewma_var)
                self.kelp_last_prices = trader_data.get("kelp_last_prices", self.kelp_last_prices)
                self.timeframe_data = trader_data.get("timeframe_data", self.timeframe_data)
            except:
                # Handle case where traderData isn't valid
                pass

        # Handle RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], 
                self.resin_fair_value, 
                2,  # width
                resin_position, 
                20  # position limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        # Handle KELP with mean reversion strategy
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_mean_reversion_strategy(state.order_depths["KELP"], state)
            result["KELP"] = kelp_orders

        # Handle SQUID_INK if needed
        # if "SQUID_INK" in state.order_depths:
        #     # Add SQUID_INK strategy here
        #     pass

        # Save state for next iteration
        trader_data = {
            "kelp_price_history": self.kelp_price_history,
            "ewma_mean": self.ewma_mean,
            "ewma_var": self.ewma_var,
            "kelp_last_prices": self.kelp_last_prices,
            "timeframe_data": self.timeframe_data
        }
        
        # Return results, conversions, and trader data
        logger.flush(state, result, 0, trader_data)
        return result, 0, jsonpickle.encode(trader_data)