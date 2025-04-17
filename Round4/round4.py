from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):

        # RAINFOREST_RESIN parameters
        self.resin_fair_value = 10000

        # DJEMBES parameters
        self.djembe_theoretical_prices = []
        self.djembe_actual_prices = []
        self.djembes_basket_history = []

        # commodity parameters
        self.position_limits = {
            "KELP": 50,
            "SQUID_INK": 50,
            "JAMS": 350,
            'CROISSANTS': 250,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
        }
        self.base_spreads = {
            "KELP": 1.3,
            "SQUID_INK": 1.1,
            "JAMS": 1.5,
            'CROISSANTS': 1,
            "VOLCANIC_ROCK": 1.2,
            "VOLCANIC_ROCK_VOUCHER_9500": 1.2,
            "VOLCANIC_ROCK_VOUCHER_9750": 1.2,
            "VOLCANIC_ROCK_VOUCHER_10250": 1.2,
            "VOLCANIC_ROCK_VOUCHER_10500": 1.2,
            "VOLCANIC_ROCK_VOUCHER_10000": 1.2,
        }
        self.inventory_factor = {
            "CROISSANTS": -0.7,
            "KELP": -0.2,
            "SQUID_INK": -0.3,
            "JAMS": -0.25,
            "VOLCANIC_ROCK": -0.7,
            "VOLCANIC_ROCK_VOUCHER_9500": -0.5,
            "VOLCANIC_ROCK_VOUCHER_9750": -0.5,
            "VOLCANIC_ROCK_VOUCHER_10250": -0.5,
            "VOLCANIC_ROCK_VOUCHER_10500": -0.5,
            "VOLCANIC_ROCK_VOUCHER_10000": -0.5,
        }
        self.momentum_factor = {
            "KELP": 0.1,
            "SQUID_INK": 0.65,
            "JAMS": 0.5,
            'CROISSANTS': 0.25,
            "VOLCANIC_ROCK": 0.1,
            "VOLCANIC_ROCK_VOUCHER_9500": 0.2,
            "VOLCANIC_ROCK_VOUCHER_9750": 0.2,
            "VOLCANIC_ROCK_VOUCHER_10250": 0.2,
            "VOLCANIC_ROCK_VOUCHER_10500": 0.2,
            "VOLCANIC_ROCK_VOUCHER_10000": 0.2,
        }
        self.ewma_mean = {
            "KELP": None,
            "SQUID_INK": None,
            "JAMS": None,
            "CROISSANTS": None,
            "VOLCANIC_ROCK": None,
            "VOLCANIC_ROCK_VOUCHER_9500": None,
            "VOLCANIC_ROCK_VOUCHER_9750": None,
            "VOLCANIC_ROCK_VOUCHER_10250": None,
            "VOLCANIC_ROCK_VOUCHER_10500": None,
            "VOLCANIC_ROCK_VOUCHER_10000": None,
        }
        self.ewma_var = {
            "KELP": None,
            "SQUID_INK": None,
            "JAMS": None,
            "CROISSANTS": None,
            "VOLCANIC_ROCK": None,
            "VOLCANIC_ROCK_VOUCHER_9500": None,
            "VOLCANIC_ROCK_VOUCHER_9750": None,
            "VOLCANIC_ROCK_VOUCHER_10250": None,
            "VOLCANIC_ROCK_VOUCHER_10500": None,
            "VOLCANIC_ROCK_VOUCHER_10000": None,
        }
        self.price_history = {
            "KELP": [],
            "SQUID_INK": [],
            "JAMS": [],
            "CROISSANTS": [],
            "VOLCANIC_ROCK": [],
            "VOLCANIC_ROCK_VOUCHER_9500": [],
            "VOLCANIC_ROCK_VOUCHER_9750": [],
            "VOLCANIC_ROCK_VOUCHER_10250": [],
            "VOLCANIC_ROCK_VOUCHER_10500": [],
            "VOLCANIC_ROCK_VOUCHER_10000": [],

        }
        self.last_prices = {
            "KELP": [],
            "SQUID_INK": [],
            "JAMS": [],
            "CROISSANTS": [],
            "VOLCANIC_ROCK": [],
            "VOLCANIC_ROCK_VOUCHER_9500": [],
            "VOLCANIC_ROCK_VOUCHER_9750": [],
            "VOLCANIC_ROCK_VOUCHER_10250": [],
            "VOLCANIC_ROCK_VOUCHER_10500": [],
            "VOLCANIC_ROCK_VOUCHER_10000": [],
        }
        self.recent_spot_prices = {
            "VOLCANIC_ROCK": [0] * 3,
            "VOLCANIC_ROCK_VOUCHER_9500": [0] * 3,
            "VOLCANIC_ROCK_VOUCHER_9750": [0] * 3,
            "VOLCANIC_ROCK_VOUCHER_10000": [0] * 3,
            "VOLCANIC_ROCK_VOUCHER_10250": [0] * 3,
            "VOLCANIC_ROCK_VOUCHER_10500": [0] * 3,
        }
        self.vouchers = {
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.time_to_expiry = 1.0  # scale 1.0 = start, 0.0 = expiry
    
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

        # mispricing
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]

            if best_ask < fair_value:
                distance = fair_value - best_ask
                multiplier = 1 + int(distance)
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
        # risk management
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit,
            product, buy_order_volume, sell_order_volume, fair_value, 1
        )
        # smart posting passive strat
        spread = baaf - bbbf if baaf and bbbf else 5
        # only if spread indicates
        if spread >= 2:
            buy_qty = position_limit - (position + buy_order_volume)
            if buy_qty > 0:
                price = bbbf + 1
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
        volatility_spread_multiplier = 1.7
        lookback_window = 100
        ewma_alpha = 0.2
        momentum_window = 5

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.price_history[product].append(mid_price)

        if len(self.price_history[product]) > lookback_window:
            self.price_history[product].pop(0)
        self.ewma_mean[product], self.ewma_var[product] = self.update_ewma(
            mid_price, self.ewma_mean[product], self.ewma_var[product], ewma_alpha
        )

        fair_value = self.ewma_mean[product] if self.ewma_mean[product] else mid_price
        std_dev = math.sqrt(self.ewma_var[product]) if self.ewma_var[product] and self.ewma_var[product] > 0 else 1

        inventory_skew = self.inventory_factor[product] * position / position_limit
        self.last_prices[product].append(mid_price)
        if len(self.last_prices[product]) > momentum_window:
            self.last_prices[product].pop(0)
        momentum = self.last_prices[product][-1] - self.last_prices[product][0] if len(self.last_prices[product]) >= 2 else 0
        momentum_skew = self.momentum_factor[product] * (momentum / std_dev)
        total_skew = inventory_skew + momentum_skew

        base_spread = self.base_spreads[product]
        dynamic_spread = base_spread + std_dev * volatility_spread_multiplier

        quote_bid = int(fair_value - dynamic_spread / 2 + total_skew)
        quote_ask = int(fair_value + dynamic_spread / 2 + total_skew)
        bid_volume = min(10, position_limit - position)
        ask_volume = min(10, position_limit + position)
        if bid_volume > 0:
            orders.append(Order(product, quote_bid, bid_volume))
        if ask_volume > 0:
            orders.append(Order(product, quote_ask, -ask_volume))
        return orders
    def croissant_strategy(self, order_depth: OrderDepth, width: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = "CROISSANTS"
        fair_value = 4270  # Mean reversion center
        position_limit = 250
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]

        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())

            # Buy if ask is significantly below mean
            if best_ask < fair_value:
                qty = 1
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_order_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            # Sell if bid is significantly above mean
            if best_bid > fair_value:
                qty = 1
                orders.append(Order(product, best_bid, -qty))
                sell_order_volume += qty

        # Risk management to avoid overshooting position limits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit,
            product, buy_order_volume, sell_order_volume, fair_value, 1
        )

        # Post passive orders if spread is wide enough
        try:
            best_ask_above_fv = min([p for p in order_depth.sell_orders if p > fair_value + 1])
            best_bid_below_fv = max([p for p in order_depth.buy_orders if p < fair_value - 1])
        except ValueError:
            best_ask_above_fv = fair_value + 3
            best_bid_below_fv = fair_value - 3

        spread = best_ask_above_fv - best_bid_below_fv
        if spread >= 2:
            buy_qty = 1
            if buy_qty > 0:
                passive_buy_price = best_bid_below_fv + 1
                orders.append(Order(product, passive_buy_price, buy_qty))

            sell_qty = 1
            if sell_qty > 0:
                passive_sell_price = best_ask_above_fv - 1
                orders.append(Order(product, passive_sell_price, -sell_qty))

        return orders

    def djembes_basket_comparison_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        product = "DJEMBES"
        position = state.position.get(product, 0)
        position_limit = 60
        
        # Ensure that both basket order depths and DJEMBES order depth are available
        if ("PICNIC_BASKET1" not in state.order_depths or
            "PICNIC_BASKET2" not in state.order_depths or
            product not in state.order_depths):
            return orders
        
        b1_depth = state.order_depths["PICNIC_BASKET1"]
        b2_depth = state.order_depths["PICNIC_BASKET2"]
        dj_depth = state.order_depths[product]
        
        # Check for necessary bid/ask data
        if (not b1_depth.buy_orders or not b1_depth.sell_orders or
            not b2_depth.buy_orders or not b2_depth.sell_orders or
            not dj_depth.buy_orders or not dj_depth.sell_orders):
            return orders
        
        # Compute mid prices for both baskets
        b1_mid = (max(b1_depth.buy_orders.keys()) + min(b1_depth.sell_orders.keys())) / 2
        b2_mid = (max(b2_depth.buy_orders.keys()) + min(b2_depth.sell_orders.keys())) / 2
        
        # Calculate normalized difference between Basket1 and Basket2
        normalized_diff = (b1_mid - b2_mid) / b2_mid
        
        # Maintain a rolling window of the normalized differences
        self.djembes_basket_history.append(normalized_diff)
        window = 1000
        if len(self.djembes_basket_history) > window:
            self.djembes_basket_history.pop(0)
        
        # Compute the mean and standard deviation of the normalized difference history
        mean_diff = np.mean(self.djembes_basket_history)
        std_diff = np.std(self.djembes_basket_history) + 1e-9  # avoid division by zero
        z_score = (normalized_diff - mean_diff) / std_diff
        
        # Access DJEMBES best bid and ask
        dj_best_bid = max(dj_depth.buy_orders.keys())
        dj_best_ask = min(dj_depth.sell_orders.keys())
        
        # Define base order size and threshold for trading signals
        order_size = int (1 * z_score)
        threshold = 0.0
        
        # Trading logic based on the z-score from the basket difference:
        # If z_score > threshold, Basket1 is relatively expensive, suggesting DJEMBES (present only in Basket1) may be overvalued -> sell DJEMBES
        # If z_score < -threshold, Basket1 is relatively cheap, suggesting DJEMBES may be undervalued -> buy DJEMBES
        if z_score > threshold and len(self.djembes_basket_history) > 20:
            if position > -position_limit:
                sell_qty = min(order_size, position_limit + position)
                # Sell slightly below the best ask
                orders.append(Order(product, dj_best_ask, -sell_qty))
        elif z_score < -threshold  and len(self.djembes_basket_history) > 20:
            if position < position_limit:
                buy_qty = min(order_size, position_limit - position)
                # Buy slightly above the best bid
                orders.append(Order(product, dj_best_bid, buy_qty))
        
        return orders

    def picnic_basket_1_strategy(self, order_depth: OrderDepth, state: TradingState):
        orders = []

        # 1. Fair price estimation for the basket
        croissant_bid = max(state.order_depths["CROISSANTS"].buy_orders.keys())
        croissant_ask = min(state.order_depths["CROISSANTS"].sell_orders.keys())
        croissant_price = (croissant_ask + croissant_bid) / 2

        jam_bid = max(state.order_depths["JAMS"].buy_orders.keys())
        jam_ask = min(state.order_depths["JAMS"].sell_orders.keys())
        jam_price = (jam_ask + jam_bid) / 2

        djembe_bid = max(state.order_depths["DJEMBES"].buy_orders.keys())
        djembe_ask = min(state.order_depths["DJEMBES"].sell_orders.keys())
        djembe_price = (djembe_ask + djembe_bid) / 2

        fair_basket_price = (6 * croissant_price) + (3 * jam_price) + djembe_price

        # 2. Market data for basket
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        position = state.position.get("PICNIC_BASKET1", 0)
        position_limit = 60  # adjust if needed

        threshold = 0.001  # ~0.1%
        max_aggressive_qty = 5
        max_passive_qty = 2

        # 3. Aggressive trades
        if best_ask < fair_basket_price * (1 - threshold) and position + max_aggressive_qty <= position_limit:
            orders.append(Order("PICNIC_BASKET1", best_ask, int(max_aggressive_qty)))

        if best_bid > fair_basket_price * (1 + threshold) and position - max_aggressive_qty >= -position_limit:
            orders.append(Order("PICNIC_BASKET1", best_bid, -int(max_aggressive_qty)))

        # 4. Passive orders
        spread_pct = 0.001
        spread = fair_basket_price * spread_pct

        passive_buy_price = fair_basket_price - spread
        passive_sell_price = fair_basket_price + spread

        # Round prices if needed (comment out if prices can be floats)
        passive_buy_price = round(passive_buy_price)
        passive_sell_price = round(passive_sell_price)
        passive_quantity = max(1, min(5, int(10 / (spread_pct * fair_basket_price))))
        if position + max_passive_qty <= position_limit:
            orders.append(Order("PICNIC_BASKET1", passive_buy_price, int(passive_quantity)))

        if position - max_passive_qty >= -position_limit:
            orders.append(Order("PICNIC_BASKET1", passive_sell_price, -int(passive_quantity)))

        return orders

    def picnic_basket_2_strategy(self, order_depth: OrderDepth, state: TradingState):
        orders = []

        # === Step 1: Compute Fair Basket Price ===
        croissant_best_bid = max(state.order_depths["CROISSANTS"].buy_orders.keys())
        croissant_best_ask = min(state.order_depths["CROISSANTS"].sell_orders.keys())
        croissant_price = (croissant_best_ask + croissant_best_bid) / 2

        jam_best_bid = max(state.order_depths["JAMS"].buy_orders.keys())
        jam_best_ask = min(state.order_depths["JAMS"].sell_orders.keys())
        jam_price = (jam_best_ask + jam_best_bid) / 2

        # Modify these weights based on actual basket composition
        expected_basket_price = (4 * croissant_price) + (2 * jam_price)

        # === Step 2: Active Trading Logic ===
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        threshold = 0.0005  # 0.1% edge required to trade actively
        quantity = 5       # Base quantity (dynamic logic added later)

        if best_ask < expected_basket_price * (1 - threshold):
            orders.append(Order("PICNIC_BASKET2", best_ask, quantity))
        elif best_bid > expected_basket_price * (1 + threshold):
            orders.append(Order("PICNIC_BASKET2", best_bid, -quantity))

        # === Step 3: Passive Trading Logic ===
        spread_pct = 0.002  # Spread is 0.1% of fair price
        spread = expected_basket_price * spread_pct

        passive_buy_price = expected_basket_price - spread
        passive_sell_price = expected_basket_price + spread

        # Use a small dynamic quantity for passive orders
        passive_quantity = max(1, min(5, int(10 / (spread_pct * expected_basket_price))))

        # Convert prices to int (if tick size is 1)
        orders.append(Order("PICNIC_BASKET2", int(passive_buy_price), passive_quantity))
        orders.append(Order("PICNIC_BASKET2", int(passive_sell_price), -passive_quantity))

        return orders

    def volcanic_voucher_strategy(self, voucher: str, strike: int, state: TradingState) -> List[Order]:
        orders = []

        spot_product = "VOLCANIC_ROCK"
        if spot_product not in state.order_depths:
            return orders

        best_bid = max(state.order_depths[spot_product].buy_orders.keys(), default=None)
        best_ask = min(state.order_depths[spot_product].sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None:
            return orders

        spot_price = (best_bid + best_ask) / 2

        u = 0.03  # 3% up move in spot
        time_value = 100
        max_order_size = 5
        max_voucher = 200

        if voucher not in state.order_depths:
            return []

        depth = state.order_depths[voucher]
        v_bid = max(depth.buy_orders.keys(), default=None)
        v_ask = min(depth.sell_orders.keys(), default=None)
        if v_bid is None or v_ask is None:
            return []

        market_price = (v_bid + v_ask) / 2
        position = state.position.get(voucher, 0)

        # Fair price from binomial estimate
        S_up = spot_price * (1 + u)
        theo = max(0, 0.5 * (S_up - strike)) + time_value
        edge = theo - market_price
        edge_threshold = market_price * 0.015  # 1.5% of market price
        # Buy undervalued
        if edge > edge_threshold and position < max_voucher:
            available_qty = depth.sell_orders[v_ask]
            qty = min(available_qty, max_order_size, max_voucher - position)
            if qty > 0 and position + qty < max_voucher:
                orders.append(Order(voucher, v_ask, qty))
                position += qty

        # Sell overvalued
        elif edge < -edge_threshold and position > -max_voucher:
            available_qty = depth.buy_orders[v_bid]
            qty = min(available_qty, max_order_size, position + max_voucher)
            if qty > 0 and position - qty > -max_voucher:
                orders.append(Order(voucher, v_bid, -qty))
                position -= qty

        # Passive orders (only place if still within bounds)
        else:
            # Passive buy
            if position + max_order_size < max_voucher:
                buy_price = int(theo - 100)
                orders.append(Order(voucher, buy_price, max_order_size))
                position += max_order_size

            # Passive sell
            if position - max_order_size > -max_voucher:
                sell_price = int(theo + 100)
                orders.append(Order(voucher, sell_price, -max_order_size))
                position -= max_order_size

        return orders
    
    def sigmoid(self, x):
        x = max(min(x, 100), -100)  # Clamp x to [-100, 100]
        return 1 / (1 + math.exp(-x))

    def premium_voucher_strategy(self, voucher: str, strike: int, state: TradingState) -> List[Order]:
        orders = []
        spot_product = "VOLCANIC_ROCK"
        window = 20
        time_to_expiry = getattr(self, "time_to_expiry", 1.0)  # scale 1.0 = start, 0.0 = expiry

        if spot_product not in state.order_depths or voucher not in state.order_depths:
            return orders

        # Get current spot price
        best_bid = max(state.order_depths[spot_product].buy_orders.keys(), default=None)
        best_ask = min(state.order_depths[spot_product].sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None:
            return orders
        spot_price = (best_bid + best_ask) / 2

        # Maintain rolling average
        if spot_product not in self.recent_spot_prices:
            self.recent_spot_prices[spot_product] = [spot_price] * window

        recent_spots = self.recent_spot_prices[spot_product]
        recent_spots.append(spot_price)
        if len(recent_spots) > window:
            recent_spots.pop(0)
        self.recent_spot_prices[spot_product] = recent_spots
        rolling_avg = sum(recent_spots) / len(recent_spots)

        # Momentum
        upward_momentum = spot_price > rolling_avg

        # Voucher price
        depth = state.order_depths[voucher]
        v_bid = max(depth.buy_orders.keys(), default=None)
        v_ask = min(depth.sell_orders.keys(), default=None)
        if v_bid is None or v_ask is None:
            return orders
        market_price = (v_bid + v_ask) / 2

        position = state.position.get(voucher, 0)
        max_voucher = 200
        max_order_size = 5

        # === 🔥 Fair price estimate with decaying time and sigmoid-like probability ===
        # payout = 400 if spot > strike else 0
        # estimate fair value = payout * probability * time_to_expiry
        volatility_factor = 10  # adjust this for how steep the sigmoid is

        # Dynamic payout: only get value if rolling average is above strike
        payout = max(0, rolling_avg - strike)

        # Estimate probability of exercising the voucher
        probability = self.sigmoid((rolling_avg - strike) / volatility_factor)

        # Final fair value estimate
        fair_price = payout * probability * time_to_expiry

        edge = fair_price - market_price

        # Buy signal
        if upward_momentum and edge > 20 and position < max_voucher:
            qty = min(depth.sell_orders[v_ask], max_order_size, max_voucher - position)
            if qty > 0:
                orders.append(Order(voucher, v_ask, qty))

        # Sell signal
        elif (not upward_momentum and edge < -20 and position > -max_voucher):
            qty = min(depth.buy_orders[v_bid], max_order_size, position + max_voucher)
            if qty > 0:
                orders.append(Order(voucher, v_bid, -qty))

        # Decay time to expiry (simple linear decay)
        self.time_to_expiry = max(0.0, self.time_to_expiry - 0.01)

        return orders
    
    def volcanic_sniper(self, state: TradingState) -> List[Order]:
        orders = []
        spot_product = "VOLCANIC_ROCK"
        max_pos = 200
        order_size = 5

        if spot_product not in state.order_depths:
            return orders

        spot_depth = state.order_depths[spot_product]
        best_bid = max(spot_depth.buy_orders.keys(), default=None)
        best_ask = min(spot_depth.sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None:
            return orders

        spot_price = (best_bid + best_ask) / 2

        for voucher, strike in self.vouchers.items():
            if voucher not in state.order_depths:
                continue

            depth = state.order_depths[voucher]
            v_bid = max(depth.buy_orders.keys(), default=None)
            v_ask = min(depth.sell_orders.keys(), default=None)
            if v_bid is None or v_ask is None:
                continue

            fair_price = (v_bid + v_ask) / 2
            position = state.position.get(voucher, 0)

            # Thresholds adjust dynamically based on proximity to strike
            proximity_factor = max(0.5, min(1.5, (strike - spot_price) / 252))
            buy_threshold = fair_price * (proximity_factor)
            sell_threshold = fair_price * (proximity_factor)

            # BUY logic
            if v_ask < buy_threshold and position < max_pos:
                qty = min(depth.sell_orders[v_ask], order_size, max_pos - position)
                if qty > 0:
                    orders.append(Order(voucher, v_ask, qty))
                    position += qty

            # SELL logic
            if v_bid > sell_threshold and position > -max_pos:
                qty = min(depth.buy_orders[v_bid], order_size, position + max_pos)
                if qty > 0:
                    orders.append(Order(voucher, v_bid, -qty))
                    position -= qty

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
            self.djembe_theoretical_prices = trader_data.get("djembe_theoretical_prices", self.djembe_theoretical_prices)
            self.djembe_actual_prices = trader_data.get("djembe_actual_prices", self.djembe_actual_prices)
            self.ewma_mean = trader_data.get("ewma_mean", self.ewma_mean)
            self.ewma_var = trader_data.get("ewma_var", self.ewma_var)
            self.price_history = trader_data.get("price_history", self.price_history)
            self.last_prices = trader_data.get("last_prices", self.last_prices)
            self.recent_spot_prices = trader_data.get("recent_spot_prices", self.recent_spot_prices)

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
            result["CROISSANTS"] = self.croissant_strategy("CROISSANTS", state.order_depths["CROISSANTS"], state)
 
        # if ("DJEMBES" in state.order_depths and "PICNIC_BASKET1" in state.order_depths and "PICNIC_BASKET2" in state.order_depths):
        #     result["DJEMBES"] = self.djembes_basket_comparison_strategy(state)
        
        # if "PICNIC_BASKET1" in state.order_depths:
        #     result["PICNIC_BASKET1"] = self.picnic_basket_1_strategy(state.order_depths["PICNIC_BASKET1"], state)
        
        # if "PICNIC_BASKET2" in state.order_depths:
        #     result["PICNIC_BASKET2"] = self.picnic_basket_2_strategy(state.order_depths["PICNIC_BASKET2"], state)
        
        # if "VOLCANIC_ROCK" in state.order_depths:
        #     result["VOLCANIC_ROCK"] = self.commodity_strategy("VOLCANIC_ROCK", state.order_depths["VOLCANIC_ROCK"], state)
        # if "VOLCANIC_ROCK_VOUCHER_9500" in state.order_depths:
        #     result["VOLCANIC_ROCK_VOUCHER_9500"] = self.premium_voucher_strategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, state) # best out of 4 strats
        # if "VOLCANIC_ROCK_VOUCHER_9750" in state.order_depths:
        #     result["VOLCANIC_ROCK_VOUCHER_9750"] = self.commodity_strategy("VOLCANIC_ROCK_VOUCHER_9750", state.order_depths["VOLCANIC_ROCK_VOUCHER_9750"], state) # best out of 4 strats
        # if "VOLCANIC_ROCK_VOUCHER_10000" in state.order_depths:
        #     result["VOLCANIC_ROCK_VOUCHER_10000"] = self.commodity_strategy("VOLCANIC_ROCK_VOUCHER_10000", state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"], state) # best out of 4 strats for it
        # if "VOLCANIC_ROCK_VOUCHER_10250" in state.order_depths:
        #     result["VOLCANIC_ROCK_VOUCHER_10250"] = self.premium_voucher_strategy("VOLCANIC_ROCK_VOUCHER_10250", 10250, state) # best out of 4 strats for it
        # if "VOLCANIC_ROCK_VOUCHER_10500" in state.order_depths:
        #     result["VOLCANIC_ROCK_VOUCHER_10500"] = self.volcanic_sniper(state) # best out of the 4 strats for it

        trader_data = {
            "djembe_theoretical_prices": self.djembe_theoretical_prices,
            "djembe_actual_prices": self.djembe_actual_prices,
            "ewma_mean": self.ewma_mean,
            "ewma_var": self.ewma_var,
            "price_history": self.price_history,
            "last_prices": self.last_prices,
            "recent_spot_prices": self.recent_spot_prices,

        }
        
        logger.flush(state, result, conversions, jsonpickle.encode(trader_data))
        # Return encoded trader data for state persistence
        return result, conversions, jsonpickle.encode(trader_data)
    