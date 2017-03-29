"""
DATA 618 - Mini-project 1: Pairs Trading
Srini Illapani
03/28/2017
"""

# Packages/libraries used
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

# Variables
stock_x = 0
stock_y = 0

# Initialization logic
def initialize(context):
    global stock_x
    global stock_y
    
    set_symbol_lookup_date('2010-01-01')
    
# 1. Create a short list of potential pairs
    
    pairs = [(symbol('CAT'), symbol('DE')), 
             (symbol('PEP'), symbol('COKE')), 
             (symbol('C'), symbol('BAC')), 
             (symbol('INFY'), symbol('WIT'))]

    # Select which pair to evaluate and assign to stock_x, stock_y
    pair = 4
   
    if pair == 1:
        stock_x, stock_y = pairs[0]
    if pair == 2:
        stock_x, stock_y = pairs[1]
    if pair == 3:
        stock_x, stock_y = pairs[2]
    if pair == 4:
        stock_x, stock_y = pairs[3]

# 3. Design rules for the trade including thresholds and any limits
    
    set_commission(commission.PerShare(cost=0.15))
    set_slippage(slippage.FixedSlippage(spread=0.25))
    
    # Set to  a custom function to schedule trades
    # to execute every day, 15 minutes after market open
    schedule_function(func=trade,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=15)
                     )
    
    # Initialize model parameters
    context.initial_setup = False
    context.entry_amount = 10000
    context.entry_threshold = 1.5
    context.exit_threshold = 0.0
    # Using a 5% Augmented Dickey Fuller test threshold
    context.adf_threshold = 0.05
    # Better performance when set to 250
    context.lookback = 250
    
    
# 2. Test the relationship of the pairs using cointegration
def check_cointegrated(x, context):
    # The pair is cointegrated if the spread is stationary
    if ts.adfuller(x)[1] < context.adf_threshold:
        return True
    else:
        return False


# Get historical data and build model
def build_model(context, data):
    global stock_x
    global stock_y

    # Get historical prices
    x_history = data.history(stock_x, 'price', context.lookback, '1d')
    y_history = data.history(stock_y, 'price', context.lookback, '1d')
    
    # OLS regression on the pair
    x = np.array(x_history)
    X = sm.add_constant(x)
    y = np.array(y_history)
    model = sm.OLS(y, X)
    results = model.fit()
    context.beta_1, context.const = results.params
    
    # Get y_hat
    y_hat = context.beta_1 + context.const * x
    
    # Get the spread
    spread = y - y_hat
    
    # Get standard deviation of spread
    context.spread_std = np.std(spread)
    
    # Call to check if pair is cointegrated 
    context.is_cointegrated = check_cointegrated(spread, context)
    
    # store relevant parameters to be used later
    context.initial_setup = True

    
# Get current spread and z-score
def get_current(context, data):
    global stock_x
    global stock_y

    current_spread = data.current(stock_y, 'price') - (context.beta_1 + context.const * data.current(stock_x, 'price'))
    current_z = current_spread / context.spread_std
    return current_spread, current_z


# Trading logic
def trade(context, data):
    global stock_x
    global stock_y
    
    if get_open_orders(stock_x) or get_open_orders(stock_y):
        return
    
    stock_x_price = data.current(stock_x, 'price')
    stock_y_price = data.current(stock_y, 'price')
    
    # We build the model for the first time
    if context.initial_setup == False:
        build_model(context, data)
    
    # Get current relationship of pair
    current_spread, current_z = get_current(context, data)
    
    # Are we above or below equilibrium?
    equilibrium = np.copysign(1, current_z)
    
    # Exit trade if the pair has reached equilibrium
    if len(context.portfolio.positions) > 0 and np.any(equilibrium != context.entry_sign or abs(current_z) < context.exit_threshold):
        order_target_percent(stock_x, 0)
        order_target_percent(stock_y, 0)
    
    # Enter a trade
    if len(context.portfolio.positions) == 0:
        
        # Rebuild the model here
        build_model(context, data)
        current_spread, current_z = get_current(context, data)
        # Are we at equilibrium?
        equilibrium = np.copysign(1, current_z)
        
        # Is pair cointegrated? Is the spread big?
        if (context.is_cointegrated and abs(current_z) >= context.entry_threshold):
            
            # Reset relationship to equilibrium as we start a trade
            context.entry_sign = equilibrium
            
            # Calculate shares to buy
            shares_x = round(context.entry_amount / stock_x_price, 0)
            shares_y = round(context.entry_amount / stock_y_price, 0)
            
            order(stock_x,      equilibrium * shares_x)
            order(stock_y, -1 * equilibrium * shares_y)