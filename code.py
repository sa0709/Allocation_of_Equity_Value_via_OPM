# =====================================================================
# IMPORTING LIBRARIES
# =====================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve   # required for solving Merton equations

# =====================================================================
# 1. READ INPUT DATA
# =====================================================================
# df1 → general model inputs (Enterprise value, r, maturity, etc.)
# df2 → cap table inputs (issue price, LP multipliers, dates, shares)
df1 = pd.read_excel('Input.xlsx', sheet_name='Input')
df2 = pd.read_excel('Input.xlsx', sheet_name='Cap_table', index_col=0)

# =====================================================================
# 2. COMPUTE EXIT AMOUNT FOR EACH SHARE CLASS
# =====================================================================

# Conversion Price formula:
#   Issue Price × LP Multiplier × (1 + Dividend Rate)^(time outstanding)
df2['Conversion Price Per Share (INR)'] = (
    df2['Issue Price Per Share (INR)']
    * df2['Liquidation Multiplier']
    * ((1 + df2['Dividend Rate']) ** (
        (df1.iloc[-1, -1] - df2['Investment Date']).dt.days / 365
    ))   # time in years
)

# Exit amount for each class (converted to million INR)
df2['Exit Amount (in Million)'] = (
    df2['Conversion Price Per Share (INR)'] * df2['# Shares'] / 10**6
)

# =====================================================================
# 3. PREPARE LIST OF UNIQUE CONVERSION PRICES
# =====================================================================

# Unique conversion prices sorted → used to create tranches
conv_price = np.unique(df2['Conversion Price Per Share (INR)'])
conv_price.sort()

# Price differences between consecutive conversion prices
diff_conv_price = np.diff(conv_price)

# =====================================================================
# 4. FIND TOTAL SHARES AT EACH CONVERSION PRICE
# =====================================================================

n_shares = []
for i in conv_price:
    # Count shares belonging to each unique conversion price
    n = df2[df2['Conversion Price Per Share (INR)'] == i]['# Shares'].sum()
    n_shares.append(n)

# =====================================================================
# 5. CUMULATIVE SHARE COUNT (NEEDED FOR TRANCHE WIDTHS)
# =====================================================================

sum_shares = 0.0
sum_share_arr = []

# Running cumulative sum of shares at each price level
for i in n_shares:
    sum_shares += i
    sum_share_arr.append(sum_shares)

# =====================================================================
# 6. LIQUIDATION PREFERENCE BREAKPOINTS
# =====================================================================

# Unique LP multipliers
n_lp = np.unique(df2['Liquidation Preference'])
n_lp.sort()

# Remove the smallest LP tier (usually common shares with 1x or 0x)
n_lp_new = np.delete(n_lp, 0)

# First breakpoint: tiny epsilon to avoid log(0) in B–S
breakpoints = [1e-17]

# Add LP breakpoints = sum of LP exit amounts for each LP tier
for i in n_lp_new:
    breakpoints.append(
        df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)'].sum()
    )

# =====================================================================
# 7. CONVERSION PRICE BREAKPOINTS
# =====================================================================

# Tranche width = cumulative shares × difference in conversion price
for j in range(len(diff_conv_price)):
    d = sum_share_arr[j] * diff_conv_price[j] / 10**6  # convert to million INR
    breakpoints.append(d)

# =====================================================================
# 8. CUMULATIVE BREAKPOINT SUM (STRIKES FOR CALL OPTIONS)
# =====================================================================

breakpoint_sum = 0
breakpoint_cum = []

# Convert tranche widths into cumulative levels
for i in breakpoints:
    breakpoint_sum += i
    breakpoint_cum.append(breakpoint_sum)

# =====================================================================
# 9. MERTON MODEL: ASSET VALUE & VOLATILITY ESTIMATION
# =====================================================================

def merton_asset_value_vol(dataframe):

    df = dataframe
    asset_vol = []
    asset_val = []

    # Loop across each time-period or input row
    for i in range(len(df.index)):
        
        E, D, sigma_E, r = df.iloc[i]   # equity value, debt, equity vol, rate
        T = 1                           # assumed 1-year horizon

        # Two nonlinear equations in two unknowns: V, sigma_V
        def equations(vars):
            V, sigma_V = vars

            d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
            d2 = d1 - sigma_V * np.sqrt(T)

            # Equation 1: Black-Scholes equity value = observed equity value
            eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E

            # Equation 2: B–S equity volatility = observed equity volatility
            eq2 = (V / E) * norm.cdf(d1) * sigma_V - sigma_E

            return (eq1, eq2)
        
        # Initial guesses
        V0 = E + D
        sigmaV0 = sigma_E * E / (E + D)

        V, sigma_V = fsolve(equations, (V0, sigmaV0))

        asset_vol.append(sigma_V)
        asset_val.append(V)

    df['Asset Volatility'] = asset_vol
    df['Asset Value'] = asset_val
    return df


# Read input for Merton model
df_input_asset_vol = pd.read_excel('Input.xlsx', sheet_name='Vol', index_col=0)
df_output_asset_vol = merton_asset_value_vol(df_input_asset_vol)

# =====================================================================
# 10. BLACK–SCHOLES CALL OPTION PRICING
# =====================================================================

def black_scholes(S, X, T, r, sigma):
    """
    Computes the Black-Scholes price of a Call option.
    S = current enterprise value (EV)
    X = strike (cumulative breakpoint)
    T = time to maturity
    r = risk-free rate
    sigma = asset volatility
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

# =====================================================================
# 11. COMPUTE OPTION VALUE FOR EACH BREAKPOINT
# =====================================================================

opt_val = []
for i in breakpoint_cum:
    opt_val.append(
        black_scholes(
            df1.iloc[0, 1],  # enterprise value
            i,              # strike
            df1.iloc[3, 1], # T
            df1.iloc[1, 1], # r
            df_output_asset_vol['Asset Volatility'].median()
        )
    )

# =====================================================================
# 12. COMPUTE TRANCHE VALUES (OCT METHOD)
# =====================================================================

# Difference between option values = value of tranche
tranche_val = np.array(np.diff(opt_val) * -1)

# Last tranche gets the residual EV
tranche_val = np.append(tranche_val, opt_val[-1])

# =====================================================================
# 13. DISTRIBUTION MATRIX SETUP
# =====================================================================

df_dist_val = pd.DataFrame(index=df2.index)   # raw allocations
df_dist = pd.DataFrame(index=df2.index)       # cumulative allocations

# Allocate LP tranches
for i in n_lp_new:
    col = 'Tranche ' + i.astype(str)
    df_dist_val[col] = df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)']
    df_dist_val.fillna(0, inplace=True)
    df_dist[col] = df_dist_val[col]

# Allocate conversion tranches
for i in range(len(conv_price)):

    tranche_num = i + np.max(n_lp_new) + 1
    col = 'Tranche ' + tranche_num.astype(str)

    # Raw allocation = shares in this bucket
    df_dist_val[col] = df2[df2['Conversion Price Per Share (INR)'] == conv_price[i]]['# Shares']
    df_dist_val.fillna(0, inplace=True)

    # Cumulative shares for waterfall
    df_dist[col] = df_dist_val.iloc[:, (np.max(n_lp_new)): tranche_num].sum(axis=1)
    df_dist.fillna(0, inplace=True)

# =====================================================================
# 14. NORMALIZE INTO PERCENTAGE ALLOCATIONS
# =====================================================================

norm_df_dist = df_dist.div(df_dist.sum())
arr = norm_df_dist.to_numpy()

# =====================================================================
# 15. COMPUTE VALUE ALLOCATION TO SHARE CLASSES
# =====================================================================

val_shares = np.dot(arr, tranche_val)

df_val = pd.DataFrame(index=norm_df_dist.index)
df_val['Total Value'] = val_shares
df_val['# Shares'] = df2['# Shares']

# Convert to per-share value (in INR)
df_val['Value Per Share'] = df_val['Total Value'] / df2['# Shares'] * 10**6

# =====================================================================
# 16. RECORD BREAKPOINTS
# =====================================================================

df_breakpoints = pd.DataFrame(index=norm_df_dist.columns)
df_breakpoints['Beginning Tranche Breakpoint'] = breakpoint_cum
df_breakpoints['Ending Tranche Breakpoint'] = df_breakpoints['Beginning Tranche Breakpoint'].shift(-1)
df_breakpoints = df_breakpoints.replace(to_replace=df_breakpoints.iloc[-1, -1], value="Infinity")

# =====================================================================
# 17. CREATE VALUE MATRIX FOR OUTPUT
# =====================================================================

df_val_1 = norm_df_dist.transpose()
df_val_1.insert(loc=0, column="Option Value", value=tranche_val)

# Multiply weights × option value to get allocated value
for col in df_val_1.columns:
    if col != 'Option Value':
        df_val_1[col] *= df_val_1['Option Value']

# Combine with summary
df_val_1_comb = pd.concat([df_val_1, df_val.transpose()])

# =====================================================================
# 18. SAVE OUTPUT TO EXCEL
# =====================================================================

excel_file_path = 'Output.xlsx'
with pd.ExcelWriter(excel_file_path) as writer:
    norm_df_dist.transpose().to_excel(writer, sheet_name='Distribution', index=True)
    df_val_1_comb.to_excel(writer, sheet_name='Value', index=True)
    df_breakpoints.to_excel(writer, sheet_name='Breakpoints', index=True)
    df_output_asset_vol.to_excel(writer, sheet_name='Asset Volatility', index=True)
