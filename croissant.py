from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, OwnTrade
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
from collections import deque
# round 4
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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()
from collections import deque
from typing import List

class Trader:
    def __init__(self):
        self.mid_prices = deque(maxlen=3)  # Only need the last 3 prices
        self.position_limit = 250
        self.prev_position_state = "long"  # can be "long", "short", or "flat"

    def croissant_strategy(self, order_depth: OrderDepth, width: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = "CROISSANTS"
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.mid_prices.append(mid_price)

        # Only act every 100 ticks = every 10,000 timestamp units
        if state.timestamp % 10000 == 0 and len(self.mid_prices) == 3:
            delta_prev = self.mid_prices[-2] - self.mid_prices[-3]
            delta_now = self.mid_prices[-1] - self.mid_prices[-2]

            def is_up(x): return x > 0
            def is_down(x): return x < 0
            def is_flat(x): return x == 0

            new_position_state = self.prev_position_state

            if is_flat(delta_now):
                if self.prev_position_state == "long":
                    new_position_state = "long"
                elif self.prev_position_state == "short":
                    new_position_state = "short"
            elif is_up(delta_now):
                if self.prev_position_state == "short":
                    new_position_state = "long"
                elif self.prev_position_state == "long":
                    new_position_state = "short"
            elif is_down(delta_now):
                if self.prev_position_state == "long":
                    new_position_state = "short"
                elif self.prev_position_state == "short":
                    new_position_state = "long"

            # Apply position change
            if new_position_state != self.prev_position_state:
                # Close current
                if position > 0:
                    orders.append(Order(product, best_bid, -position))
                elif position < 0:
                    orders.append(Order(product, best_ask, -position))

                # Open new
                if new_position_state == "long":
                    orders.append(Order(product, best_ask, self.position_limit))
                elif new_position_state == "short":
                    orders.append(Order(product, best_bid, -self.position_limit))

                self.prev_position_state = new_position_state

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except Exception:
                trader_data = {}
        
        if trader_data:
            self.mid_price = trader_data.get("mid_price", getattr(self, "mid_price", []))


        if "CROISSANTS" in state.order_depths:
            result["CROISSANTS"] = self.croissant_strategy(state.order_depths["CROISSANTS"], 2, state)

        trader_data = {
            "mid_price": getattr(self, "mid_price", 0),
        }
        logger.flush(state, result, conversions, jsonpickle.encode(trader_data))
        # Return encoded trader data for state persistence
        return result, conversions, jsonpickle.encode(trader_data)