from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

class Trader:
    def __init__(self):
        # JAMS parameters
        self.jam_position_limit = 350

        # CROISSANTS parameters
        self.croissant_position_limit = 250
        self.croissant_prices = []
        self.rolling_window = 20
        self.momentum_window = 5



        # DJEMBE parameters
        self.djembe_position_limit = 60

        # PICNIC_BASKET_1 parameters
        # contain 6 croissants, 3 jams, 1 djembe
        self.picnic_basket_1_limit = 60

        # PICNIC_BASKET_2 parameters
        # contain 4 croissants, 2 jams
        self.picnic_basket_2_limit = 100

        # Conversion tracking
        self.prev_timestamp = None
        self.timeframe_data = {}
   
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

    def run(self, state: TradingState):
        result = {}

        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except Exception:
                trader_data = {}

        if trader_data:
            # update parameters
            pass

        # if "JAMS" in state.order_depths:
        #     result["JAMS"] = self.jams_strategy(state.order_depths["JAMS"], state)

        if "CROISSANTS" in state.order_depths:
            result["CROISSANTS"] = self.croissants_strategy(state.order_depths["CROISSANTS"], state)

        # if "DJEMBE" in state.order_depths:
        #     result["DJEMBE"] = self.djembe_strategy(state.order_depths["DJEMBE"], state)

        trader_data = {
            "timeframe_data": self.timeframe_data
        }

        return result, 0, jsonpickle.encode(trader_data)