from enum import Enum, auto

import numpy as np
import polars as pl
import QuantLib as ql


class CurveMode(Enum):
    FLAT = auto()
    FLATINCEPTION = auto()  # theta(t) & r(t) force forward/zero coupon curve evolution.
    ROLLUP = auto()
    ROLLDOWN = auto()


# TODO dataclass
class ModelParam:
    def __init__(
        self,
        r: float,
        curve_mode: CurveMode,
        a: float | None = None,
        sigma: float | None = None,
        grid_points: int | None = None,
    ) -> None:
        self.r = r
        self.a = a if a is not None else 0.2
        self.curve_mode = curve_mode
        self.sigma = sigma if sigma is not None else 0.05
        self.grid_points = grid_points if grid_points is not None else 100


def get_zc_bond() -> ql.FixedRateBond:
    settlement_days = 2
    face_amount = 100

    issue_date = ql.Date(30, 6, 2019)
    maturity_date = ql.Date(15, 9, 2023)
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


def calc_zc_bond(
    bond: ql.Bond, d: ql.Date, short_rate: float, model_param: ModelParam
) -> dict:
    ql.Settings.instance().evaluationDate = d
    settle_date = bond.calendar().advance(d, bond.settlementDays(), ql.Days)

    dates = ql.MakeSchedule(
        d,
        bond.maturityDate(),  # maturity date will be skipped
        ql.Period("1D"),
        calendar=bond.calendar(),
    )
    # dates = list(dates)
    dates = [bond.calendar().advance(x, bond.settlementDays(), ql.Days) for x in dates]

    r = model_param.r
    a = model_param.a
    sigma = model_param.sigma
    if model_param.curve_mode == CurveMode.FLAT:
        # This formula is taken from "Flat at Inception" by setting t_0=t (PDE rolls)
        # note: if short_rate == r this reduces to flat curve
        zero_rates = [short_rate] + [
            r
            + (1.0 / a)
            * (short_rate - r)
            * (np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0)))
            / ((d_ - settle_date) / 360.0)
            for d_ in dates[1:]
        ]
        prev_settle_date = bond.calendar().advance(
            d, bond.settlementDays() - 1, ql.Days
        )
        prev_zero_rates = [r] + [
            r
            + (
                (1.0 / a)
                * (short_rate - r)
                * (np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0)))
                + ((sigma**2) / (4.0 * a**3))
                * (1.0 - np.exp(-2.0 * a * (settle_date - prev_settle_date) / 360.0))
                * np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
            )
            / ((d_ - settle_date) / 360.0)
            for d_ in dates[1:]
        ]
        theta0 = a * r
    elif model_param.curve_mode == CurveMode.FLATINCEPTION:
        # zero rates consistent with flat term structure as of bond issue date
        # (!) for settle_date = bond issue date, this is flat structure but generally not so
        zero_rates = [r] + [
            r
            + (
                (1.0 / a)
                * (short_rate - r)
                * (np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0)))
                + ((sigma**2) / (4.0 * a**3))
                * (1.0 - np.exp(-2.0 * a * (settle_date - bond.issueDate()) / 360.0))
                * np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0))
            )
            / ((d_ - settle_date) / 360.0)
            for d_ in dates[1:]
        ]
        prev_zero_rates = zero_rates
        theta0 = a * r + sigma**2 / (2 * a) * (
            1.0 - np.exp(-2.0 * a * (settle_date - bond.issueDate()) / 360.0)
        )
    elif (
        model_param.curve_mode == CurveMode.ROLLUP
    ):  # this curve is "lifting up" over time
        # forwards = [flat_rate] + [flat_rate - (sigma**2)/(2*a**2)*np.square(1. - np.exp(-a*(d_-d)/360)) for d_ in dates[1:]]
        # curve = ql.ForwardCurve(list(dates), forwards, ql.Actual360(), bond.calendar()) # this doesn't admit semiannual compounding
        zero_rates = [r] + [
            r
            - ((sigma**2) / (2.0 * a**2))
            * (
                (1.0 / a)
                * (short_rate - r)
                * (np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0)))
                + (
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
        theta0 = a * r
    elif (
        model_param.curve_mode == CurveMode.ROLLDOWN
    ):  # this curve is "shifting down" over time (intuitive situation)
        zero_rates = [r] + [
            r
            + (
                (1.0 / a)
                * (short_rate - r)
                * (np.square(1.0 - np.exp(-a * (d_ - settle_date) / 360.0)))
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
        theta0 = a * r + sigma**2 / a
    else:
        prev_zero_rates = zero_rates = []
        theta0 = 0.0
        NotImplementedError(f"Unrecognized curve_mode: {model_param.curve_mode}")

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
            ddays=(
                pl.col("settle_date").shift(-1) - pl.col("settle_date")
            ).dt.total_days(),
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
            pnl_oas=pl.col("dv_dt_oas").mul(pl.col("ddays")).truediv(360.0),
            pnl_dur=pl.col("dv_dt_dur").mul(pl.col("ddays")).truediv(360.0),
            pnl_conv=pl.col("dv_dt_conv").mul(pl.col("ddays")).truediv(360.0),
            pnl_roll=(pl.col("price") - pl.col("rolled_price")).shift(-1),
        )
        .with_columns(
            pnl_oas_full=pl.col("pnl_oas")
            + pl.col("pnl_dur")
            + pl.col("pnl_conv")
            + pl.col("pnl_roll"),
            pnl_err=(
                pl.col("pnl_oas")
                + pl.col("pnl_dur")
                + pl.col("pnl_conv")
                + pl.col("pnl_roll")
            ).sub(pl.col("pnl")),
        )
    )

    res = res.with_columns(
        cpnl=pl.col("pnl").cum_sum(),
        cpnl_oas=pl.col("pnl_oas").cum_sum(),
        cpnl_dur=pl.col("pnl_dur").cum_sum(),
        cpnl_conv=pl.col("pnl_conv").cum_sum(),
        cpnl_roll=pl.col("pnl_roll").cum_sum(),
        cpnl_oas_full=pl.col("pnl_oas_full").cum_sum(),
    )
    return res
