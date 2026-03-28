import datetime

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import polars as pl
import QuantLib as ql


class CurveMode(Enum):
    FLAT = auto()
    FLATINCEPTION = auto()  # theta(t) & r(t) force forward/zero coupon curve evolution.
    ROLLUP = auto()
    ROLLDOWN = auto()


@dataclass(slots=True)
class ModelParam:
    r: float
    curve_mode: CurveMode
    a: float = 0.2
    sigma: float = 0.05
    grid_points: int = 100


class ZeroCouponBondConfig:
    def __init__(
        self,
        settlement_days: int | None = None,
        face_amount: float | None = None,
        issue_date: datetime.date | None = None,
        maturity_date: datetime.date | None = None,
    ):
        self.settlement_days = settlement_days
        self.face_amount = face_amount
        self.issue_date = issue_date
        self.maturity_date = maturity_date


def _gen_rate_curves(
    dates: list[ql.Date],
    short_rate: float,
    model_param: ModelParam,
    inception_date: ql.Date | None = None,
):
    settle_date = dates[0]
    r = model_param.r
    a = model_param.a
    sigma = model_param.sigma
    match model_param.curve_mode:
        case CurveMode.FLAT:
            # This formula is taken from "Flat at Inception" by setting t_0=t (PDE rolls)
            # note: if short_rate == r this reduces to flat curve
            zero_rates = [short_rate] + [
                r
                + (1.0 / a)
                * (short_rate - r)
                * (1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                / ((d_ - settle_date) / 360.0)
                for d_ in dates[1:]
            ]
            prev_zero_rates = [short_rate] + [
                r
                + (
                    (1.0 / a)
                    * (short_rate - r)
                    * (1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                    + ((sigma**2) / (4.0 * a**3))
                    * (1.0 - np.exp(-2.0 * a * (settle_date - inception_date) / 360.0))
                    * np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                )
                / ((d_ - settle_date) / 360.0)
                for d_ in dates[1:]
            ]
        case CurveMode.FLATINCEPTION:
            # zero rates consistent with flat term structure as of bond issue date
            # (!) for settle_date = bond issue date, this is flat structure but generally not so
            zero_rates = [short_rate] + [
                r
                + (
                    (1.0 / a)
                    * (short_rate - r)
                    * (1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                    + ((sigma**2) / (4.0 * a**3))
                    * (1.0 - np.exp(-2.0 * a * (settle_date - inception_date) / 360.0))
                    * np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                )
                / ((d_ - settle_date) / 360.0)
                for d_ in dates[1:]
            ]
            prev_zero_rates = zero_rates
        case CurveMode.ROLLUP:
            # this curve is "lifting up" over time
            # forwards = [flat_rate] + [flat_rate - (sigma**2)/(2*a**2)*np.square(1. - np.exp(-a*(d_-d)/360)) for d_ in dates[1:]]
            # curve = ql.ForwardCurve(list(dates), forwards, ql.Actual360(), bond.calendar()) # this doesn't admit semiannual compounding
            zero_rates = [short_rate] + [
                r
                + (
                    (1.0 / a)
                    * (short_rate - r)
                    * (1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                    - ((sigma**2) / (2.0 * a**2))
                    * (
                        ((d_ - settle_date) / 360.0)
                        - (2.0 / a) * (1.0 - np.exp(-a * ((d_ - settle_date) / 360.0)))
                        + (0.5 / a)
                        * (1.0 - np.exp(-2.0 * a * ((d_ - settle_date) / 360.0)))
                    )
                )
                / ((d_ - settle_date) / 360.0)
                for d_ in dates[1:]
            ]
            prev_zero_rates = zero_rates
        case CurveMode.ROLLDOWN:
            # this curve is "shifting down" over time (intuitive situation)
            zero_rates = [short_rate] + [
                r
                + (
                    (1.0 / a)
                    * (short_rate - r)
                    * (1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
                    + ((sigma**2) / (2.0 * a**2))
                    * (
                        ((d_ - settle_date) / 360.0)
                        - (0.5 / a)
                        * (1.0 - np.exp(-2.0 * a * ((d_ - settle_date) / 360.0)))
                    )
                )
                / ((d_ - settle_date) / 360.0)
                for d_ in dates[1:]
            ]
            prev_zero_rates = zero_rates
        case _:
            prev_zero_rates = zero_rates = []
            raise NotImplementedError(
                f"Unrecognized curve_mode: {model_param.curve_mode}"
            )

    return prev_zero_rates, zero_rates


def simulate_short_rate_paths(
    dates: list[ql.Date],
    short_rate: float,
    model_param: ModelParam,
    inception_date: ql.Date | None = None,
    n_paths: float | None = None,
    calendar: ql.Calendar | None = None,
    seed: int = 0,
):
    settle_date = dates[0]
    if calendar is None:
        calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
    a = model_param.a
    sigma = model_param.sigma

    _, zero_rates = _gen_rate_curves(dates, short_rate, model_param, inception_date)
    curve = ql.ZeroCurve(
        dates,
        zero_rates,
        ql.Actual360(),
        calendar,
        ql.Linear(),
        ql.Compounded,
        ql.Semiannual,
    )
    ts_handle = ql.YieldTermStructureHandle(curve)

    hw_process = ql.HullWhiteProcess(ts_handle, a, sigma)
    day_counter = ql.Actual360()
    grid = ql.TimeGrid([day_counter.yearFraction(settle_date, d_) for d_ in dates])
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(
            len(dates) - 1, ql.UniformRandomGenerator(seed=seed)
        )
    )
    gen = ql.GaussianPathGenerator(hw_process, grid, rng, False)

    return {
        "dates": dates,
        "paths": [
            list(gen.next().value()) for _ in range(1 if n_paths is None else n_paths)
        ],
    }


def get_zc_bond(bond_config: ZeroCouponBondConfig | None) -> ql.FixedRateBond:
    settlement_days = (
        2 if bond_config.settlement_days is None else bond_config.settlement_days
    )
    face_amount = 100 if bond_config.face_amount is None else bond_config.face_amount
    issue_date = ql.Date.from_date(
        bond_config.issue_date
    )  # ql.Date.from_date(30, 6, 2019)
    maturity_date = ql.Date.from_date(bond_config.maturity_date)  # ql.Date(15, 9, 2023)
    calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
    payment_convention = ql.Following

    # zcbond = ql.ZeroCouponBond(settlement_days, calendar, face_amount, maturity_date, payment_convention, face_amount, issue_date)
    # return zcbond

    czcbond = ql.CallableZeroCouponBond(
        settlement_days,
        face_amount,
        calendar,
        maturity_date,
        ql.Actual360(),
        payment_convention,
        face_amount,
        issue_date,
        ql.CallabilitySchedule(),
    )
    return czcbond


def deserialize_bond_func(func):
    def deserialized_func(*args, **kwargs):
        return func(
            *[
                get_zc_bond(a) if isinstance(a, ZeroCouponBondConfig) else a
                for a in args
            ],
            **kwargs,
        )

    return deserialized_func


@deserialize_bond_func
def calc_zc_bond(
    bond: ql.Bond | ZeroCouponBondConfig,
    d: ql.Date,
    short_rate: float,
    model_param: ModelParam,
) -> dict:
    ql.Settings.instance().evaluationDate = d
    dates = ql.MakeSchedule(
        d,
        bond.maturityDate(),  # maturity date will be skipped
        ql.Period("1D"),
        calendar=bond.calendar(),
    )
    dates = [bond.calendar().advance(x, bond.settlementDays(), ql.Days) for x in dates]
    settle_date = dates[0]

    r = model_param.r
    a = model_param.a
    sigma = model_param.sigma

    match model_param.curve_mode:
        case CurveMode.FLAT:
            theta0 = a * r
            inception_date = bond.calendar().advance(
                d, bond.settlementDays() - 1, ql.Days
            )  # rolls of prev curve
        case CurveMode.FLATINCEPTION:
            theta0 = a * r + sigma**2 / (2 * a) * (
                1.0 - np.exp(-2.0 * a * (settle_date - bond.issueDate()) / 360.0)
            )
            inception_date = bond.issueDate()
        case CurveMode.ROLLUP:
            theta0 = a * r
            inception_date = None
        case CurveMode.ROLLDOWN:
            theta0 = a * r + sigma**2 / a
            inception_date = None
        case _:
            theta0 = 0.0
            inception_date = None

    prev_zero_rates, zero_rates = _gen_rate_curves(
        dates, short_rate, model_param, inception_date
    )
    curve = ql.ZeroCurve(
        dates,
        zero_rates,
        ql.Actual360(),
        bond.calendar(),
        ql.Linear(),
        ql.Compounded,
        ql.Semiannual,
    )

    # forwards =  [curve.forwardRate(d_, d_, ql.Actual360(), ql.Compounded, ql.Semiannual).rate() for d_ in dates] # transform back
    ts_handle = ql.YieldTermStructureHandle(curve)

    # bond_engine = ql.DiscountingBondEngine(ts_handle)
    # bond.setPricingEngine(bond_engine)
    engine = ql.TreeCallableFixedRateBondEngine(
        ql.HullWhite(ts_handle, a, sigma), model_param.grid_points
    )
    bond.setPricingEngine(engine)

    price = bond.cleanPrice()
    coupon = {cf.date(): cf.amount() for cf in bond.cashflows()}.get(d, 0.0)
    effective_duration = bond.effectiveDuration(
        0.0, ts_handle, ql.Actual360(), ql.Compounded, ql.Semiannual, 5e-4
    )
    pde_duration = (
        1.0 / a
    ) * (  # note that duration has opposite sign to the derivative
        1.0 - np.exp(-a * (bond.maturityDate() - settle_date) / 360.0)
    )
    effective_convexity = bond.effectiveConvexity(
        0.0, ts_handle, ql.Actual360(), ql.Compounded, ql.Semiannual, 5e-4
    )
    pde_convexity = pde_duration**2

    # rolling the curve
    if model_param.curve_mode == CurveMode.FLAT:
        prev_curve = ql.ZeroCurve(
            dates,
            prev_zero_rates,
            ql.Actual360(),
            bond.calendar(),
            ql.Linear(),
            ql.Compounded,
            ql.Semiannual,
        )
        engine = ql.TreeCallableFixedRateBondEngine(
            ql.HullWhite(ql.YieldTermStructureHandle(prev_curve), a, sigma),
            model_param.grid_points,
        )
        bond.setPricingEngine(engine)
        rolled_price = bond.cleanPrice()
    else:
        rolled_price = price

    return {
        "date": d.to_date(),
        "settle_date": settle_date.to_date(),
        "r_theta": r,
        "a": a,
        "sigma": sigma,
        "theta0": theta0,
        "short_rate": short_rate,
        "price": price,
        "rolled_price": rolled_price,
        "coupon": coupon,
        "effective_duration": effective_duration,
        "pde_duration": pde_duration,
        "effective_convexity": effective_convexity,
        "pde_convexity": pde_convexity,
    }


def pnl_decomposition(res: pl.DataFrame) -> pl.DataFrame:
    # Check PDE
    # V_t + 0.5*s**2*V_xx(x, t) + (theta - a*x) * V_x = x*V
    # note that for the first PDE, V_x term is multiplied by 0
    freq = 2
    res = (
        res.with_columns(
            ddays=pl.col("settle_date").diff(-1).dt.total_days().neg(),
            dr=pl.col("short_rate").diff(-1).neg(),
            dv_dt_oas=pl.col("short_rate")
            .truediv(freq)
            .add(1.0)
            .log()
            .mul(freq)
            .mul("price"),
            dv_dt_dur=pl.col("pde_duration")
            .mul("price")
            .mul(pl.col("theta0") - pl.col("a").mul("short_rate")),
            # dv_dt_conv=pl.col("effective_convexity") # effective_convexity<>pde_convexity
            dv_dt_conv=pl.col("pde_convexity")
            .mul("price")
            .mul(0.05**2)
            .truediv(2.0)
            .neg(),
            pnl=(pl.col("price").shift(-1) - pl.col("price")).add(
                pl.col("coupon").shift(-1)
            ),
        )
        .with_columns(
            pnl_theta_carry=pl.col("dv_dt_oas").mul(pl.col("ddays")).truediv(360.0),
            pnl_theta_dur=pl.col("dv_dt_dur").mul(pl.col("ddays")).truediv(360.0),
            pnl_theta_conv=pl.col("dv_dt_conv").mul(pl.col("ddays")).truediv(360.0),
            pnl_dr=pl.col("pde_duration").mul("price").mul("dr").neg(),
            pnl_dr2=pl.col("pde_convexity").mul("price").mul(0.5 * pl.col("dr").pow(2)),
            pnl_model_roll=(pl.col("price") - pl.col("rolled_price")).shift(-1),
        )
        .with_columns(
            pnl_decomposition=pl.col("pnl_theta_carry")
            + pl.col("pnl_theta_dur")
            + pl.col("pnl_theta_conv")
            + pl.col("pnl_dr")
            + pl.col("pnl_dr2")
            + pl.col("pnl_model_roll")
        )
        .with_columns(pnl_err=pl.col("pnl") - pl.col("pnl_decomposition"))
    )
    # cumulative results
    res = res.with_columns(pl.selectors.starts_with("pnl").cum_sum().name.prefix("c"))

    return res


class unpack:
    def __init__(self, func):
        self.func = func
        # self.__name__ = func.__name__
        # self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[-1], (list, tuple)):
            args.extend(args.pop())
        return self.func(*args, **kwargs)


def job_func_full_(
    zcb_config: ZeroCouponBondConfig,
    model_param: ModelParam,
    short_rate: float,
    d: datetime.date,
) -> dict:
    """Utility for parallel processing"""
    try:
        return calc_zc_bond(zcb_config, ql.Date.from_date(d), short_rate, model_param)
    except RuntimeError:
        return {}


job_func_full = unpack(job_func_full_)
