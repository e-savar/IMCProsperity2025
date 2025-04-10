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

        if "JAMS" in state.order_depths:
            result["JAMS"] = self.jams_strategy(state.order_depths["JAMS"], state)

        if "CROISSANTS" in state.order_depths:
            result["CROISSANTS"] = self.croissants_strategy(state.order_depths["CROISSANTS"], state)

        if "DJEMBE" in state.order_depths:
            result["DJEMBE"] = self.djembe_strategy(state.order_depths["DJEMBE"], state)

        trader_data = {
            "timeframe_data": self.timeframe_data
        }

        return result, 0, jsonpickle.encode(trader_data)