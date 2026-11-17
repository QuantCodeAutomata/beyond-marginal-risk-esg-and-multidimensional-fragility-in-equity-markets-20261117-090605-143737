"""Parameters for the ESG / multidimensional-fragility replication.

Every module-level variable states its meaning, unit, and the paper section
or equation it comes from. Variables without a stated paper value are marked
"not specified in the paper".
"""

# ---------------------------------------------------------------------------
# Sample period and universe (Sec. 3.1)
# ---------------------------------------------------------------------------
SAMPLE_START = "2019-01-01"   # first outcome month, paper Sec. 3.1
SAMPLE_END = "2025-11-30"     # last outcome month, paper Sec. 3.1 ("through November 2025")
UNIVERSE_SOURCE = "wikipedia_sp500_current"  # paper cites "Wikipedia contributors, 2026" historical S&P 500 constituents; the live article is the only obtainable proxy for that snapshot
UNIVERSE_SIZE_PAPER = 526      # firms in the full panel, paper Sec. 3 data description
UNIVERSE_SIZE_RUN = 40         # tickers actually downloaded for the market-data demonstration (time-limited; see results/RESULTS.md)
FIRM_MONTHS_PAPER = 35445      # paper: total firm-months in the panel
REGRESSION_FIRM_MONTHS = {     # paper: common-sample firm-months per channel
    "return": 29792,
    "rvol": 30071,
    "chl": 30072,
}

# ---------------------------------------------------------------------------
# Outcome construction (Sec. 3.3, Equations 1-2)
# ---------------------------------------------------------------------------
RVOL_ANNUALIZATION_FACTOR = 252  # trading days/year used in RVOL_t = sqrt(252) * sd(daily returns in month), Eq. 2
WINSORIZE_LOWER_PCTL = 0.01       # controls and momentum winsorized at 1st percentile, Sec. 3.2
WINSORIZE_UPPER_PCTL = 0.99       # controls and momentum winsorized at 99th percentile, Sec. 3.2
ACCOUNTING_LAG_MONTHS = 6          # accounting variables lagged 6 months from fiscal statement date, data_requirements

# ---------------------------------------------------------------------------
# ESG variable (Sec. 3.2)
# ---------------------------------------------------------------------------
ESG_PRIMARY_FIELD = "IAS"          # MSCI Industry Adjusted Score, primary ESG construct, Sec. 3.2
ESG_SECONDARY_FIELD = "WAKIS"      # MSCI Weighted Average Key Issue Score, Sec. 3.2 / exp_4
ESG_POOLED_SD = 2.67               # pooled SD of raw IAS on the 3-channel common sample, Sec. 3.2
ESG_WINSORIZED = False             # ESG is raw: neither standardized nor winsorized in primary models, Sec. 3.2

# ---------------------------------------------------------------------------
# Market state (Sec. 4.1, Equation 4)
# ---------------------------------------------------------------------------
MARKET_INDEX_TICKER = "SPY"        # proxy for the S&P 500 price index return R^M_t (paper does not name a ticker/provider)
N_DOWN_MONTHS_PAPER = 27           # paper: negative-market months in the 83-month sample
N_UP_MONTHS_PAPER = 56             # paper: positive-market months in the 83-month sample
N_MONTHS_PAPER = 83                # paper: total sample months

# ---------------------------------------------------------------------------
# Ex ante exposure windows (Sec. 3.2)
# ---------------------------------------------------------------------------
BETA_WINDOW_MONTHS_PRIMARY = 24    # 24-month daily market beta, return channel, Sec. 3.2
BETA_WINDOW_MONTHS_ROBUST = 12     # 12-month beta, robustness (Table A6)
RVOL_EXPOSURE_WINDOW_MONTHS = 12   # trailing 12-month realized volatility, volatility channel
CHL_EXPOSURE_WINDOW_MONTHS = 12    # trailing 12-month log CHL spread, liquidity channel

# ---------------------------------------------------------------------------
# Joint multidimensional fragility thresholds (Sec. 3.4, Equation 3)
# ---------------------------------------------------------------------------
JOINT_THRESHOLD_PRIMARY = 0.80     # frozen pooled 80th percentile, primary Joint80 state
JOINT_THRESHOLD_GRID = (0.70, 0.75, 0.80, 0.85, 0.90)  # exp_5 threshold-sensitivity grid
LOSS_Q80_PAPER = 0.0577            # paper: frozen 80th-percentile Loss threshold (5.77%)
LOG_RVOL_Q80_PAPER = -0.973        # paper: frozen 80th-percentile log RVOL threshold (~37.8% annualized)
LOG_CHL_Q80_PAPER = -4.698         # paper: frozen 80th-percentile log CHL threshold (~0.91% spread)
JOINT80_EVENTS_PAPER = 1307        # paper: Joint80 events in the 29,792-obs regression sample (4.39%)

# ---------------------------------------------------------------------------
# Inference (Sec. 4, Appendix)
# ---------------------------------------------------------------------------
HOLM_ALPHA = 0.10                        # significance level used to judge Holm survival, Sec. 4 / Table discussion
WILD_BOOTSTRAP_REPLICATIONS = 9999        # Rademacher wild-cluster-bootstrap replications, methodology step 6
WILD_BOOTSTRAP_REPLICATIONS_RUN = 200     # reduced replication count used in the executed run (time-limited; see results/RESULTS.md); --full selects WILD_BOOTSTRAP_REPLICATIONS
WILD_BOOTSTRAP_CLUSTER_DIM = "month"      # cluster dimension for the wild bootstrap, methodology step 6
WILD_BOOTSTRAP_SEED = 20191130            # not specified in the paper; fixed for reproducibility
CLUSTER_DIMS = ("firm", "month")          # two-way clustering dimensions, Sec. 4

# ---------------------------------------------------------------------------
# Complete fitted ESG slope evaluation points (Table A3 / exp_3, exp_4)
# ---------------------------------------------------------------------------
MARKET_STATE_GRID = (-0.10, -0.05, 0.0, 0.05, 0.10)  # market-return evaluation points for complete slopes

# ---------------------------------------------------------------------------
# Synthetic-panel calibration (used only where MSCI ESG / full accounting data
# cannot be obtained from permitted sources; see results/RESULTS.md)
# ---------------------------------------------------------------------------
SYNTHETIC_SEED = 1234567                  # not specified in the paper; fixed for reproducibility
SYNTHETIC_N_FIRMS_PAPER = 526              # paper panel size, used when --full is passed
SYNTHETIC_N_FIRMS_RUN = 140                # reduced firm count for the executed run (time-limited; see results/RESULTS.md)
SYNTHETIC_IAS_WAKIS_CORR_PAPER = 0.196     # paper: overall Pearson correlation between IAS and WAKIS levels, Sec. 3.2/exp_4
SYNTHETIC_IAS_CHANGE_FREQ_PAPER = 0.0595   # paper: share of adjacent firm-month pairs with an IAS change, exp_4
SYNTHETIC_WAKIS_CHANGE_FREQ_PAPER = 0.2739  # paper: share of adjacent firm-month pairs with a WAKIS change, exp_4
