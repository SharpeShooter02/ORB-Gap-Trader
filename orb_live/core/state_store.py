"""
core/state_store.py — SQLite state persistence via SQLAlchemy Core.

All live-session state is stored here so that restarts are safe and the
full session can be audited from disk after the fact.

Schema (20 tables):
  1.  universe          — ACTIVE symbols + metadata snapshot
  2.  day_state         — per-date session control (phase, kill flag)
  3.  gap_scan          — overnight gap measurement per symbol per date
  4.  ps_filter_result  — prior-session filter outcome per symbol per date
  5.  orb_window        — ORB high/low computed at 10:00 ET per symbol
  6.  breakout_signal   — confirmed breakout per symbol per date
  7.  pending_orders    — submitted orders not yet confirmed
  8.  open_positions    — full position state (all simulate_trade fields)
  9.  closed_trades     — fully closed trade records (mirrors backtest output)
  10. equity_curve      — daily equity snapshots
  11. sigma_calibration — sigma estimates used for PS filter, with timestamp
  12. alert_log         — risk events, kill-switch triggers, anomalies
  13. system_events     — general lifecycle log (startup, shutdown, reconnects)
  14. underlying_bars   — previous-session OHLCV for PS filter computation
  15. rtg_history       — rolling per-symbol RTG values for percentile computation
  16. liquidity_metrics — ADV / DV data per symbol per check date
  17. candidates        — per-symbol per-session qualification decision log
  18. sigma_history     — audit trail of sigma recalibrations
  19. fills             — individual order fill records per leg
  20. indicator_state   — per-bar EMA snapshot for audit / restart recovery
"""

from datetime import date, datetime, timezone
UTC = timezone.utc
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer,
    MetaData, String, Table, Text, create_engine, event,
    inspect, text,
)
from sqlalchemy.engine import Engine

_metadata = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

universe = Table("universe", _metadata,
    Column("symbol",     String(16), primary_key=True),
    Column("underlying", String(16), nullable=False),
    Column("inverse",    Boolean,    nullable=False),
    Column("leverage",   Float,      nullable=False),
    Column("is_class_a", Boolean,    nullable=False, default=False),
    Column("updated_at", DateTime,   nullable=False),
)

day_state = Table("day_state", _metadata,
    Column("trade_date",     Date,    primary_key=True),
    Column("phase",          String(32), nullable=False, default="pre_market"),
    Column("kill_triggered", Boolean, nullable=False, default=False),
    Column("kill_reason",    String(256)),
    Column("n_prequalified", Integer, default=0),
    Column("n_traded",       Integer, default=0),
    Column("session_pnl",    Float,   default=0.0),
    Column("created_at",     DateTime, nullable=False),
    Column("updated_at",     DateTime, nullable=False),
)

gap_scan = Table("gap_scan", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("trade_date",   Date,    nullable=False),
    Column("symbol",       String(16), nullable=False),
    Column("prev_close",   Float),
    Column("open_price",   Float),
    Column("gap_pct",      Float),
    Column("gap_dir",      Integer),  # +1 gap-up, -1 gap-down
    Column("qualifies",    Boolean),
    Column("filter_reason",String(128)),
    Column("scanned_at",   DateTime),
)

ps_filter_result = Table("ps_filter_result", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("trade_date",   Date,    nullable=False),
    Column("symbol",       String(16), nullable=False),
    Column("underlying",   String(16)),
    Column("ul_move_pct",  Float),
    Column("threshold_pct",Float),
    Column("passed",       Boolean,  nullable=False),
    Column("checked_at",   DateTime),
)

orb_window = Table("orb_window", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("trade_date",   Date,    nullable=False),
    Column("symbol",       String(16), nullable=False),
    Column("orb_high",     Float),
    Column("orb_low",      Float),
    Column("orb_close",    Float),
    Column("orb_range_pct",Float),
    Column("computed_at",  DateTime),
)

breakout_signal = Table("breakout_signal", _metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("trade_date",    Date,    nullable=False),
    Column("symbol",        String(16), nullable=False),
    Column("direction",     Integer),   # +1 long / -1 short
    Column("breakout_price",Float),
    Column("orb_high",      Float),
    Column("orb_low",       Float),
    Column("detected_at",   DateTime),
)

pending_orders = Table("pending_orders", _metadata,
    Column("client_order_id", String(64), primary_key=True),
    Column("trade_date",   Date,     nullable=False),
    Column("symbol",       String(16), nullable=False),
    Column("side",         String(8),  nullable=False),  # buy / sell
    Column("qty",          Float,      nullable=False),
    Column("order_type",   String(16), nullable=False),
    Column("limit_price",  Float),
    Column("stop_price",   Float),
    Column("submitted_at", DateTime,   nullable=False),
    Column("broker_order_id", String(64)),
    Column("status",       String(32), default="pending"),
)

open_positions = Table("open_positions", _metadata,
    Column("symbol",            String(16), primary_key=True),
    Column("trade_date",        Date,       nullable=False),
    Column("direction",         Integer,    nullable=False),
    Column("status",            String(32), default="open"),  # entering/open/unfilled/closed
    # Entry details
    Column("entry_price",       Float,      nullable=False),  # limit price from compute_entry
    Column("actual_entry_price",Float),                       # actual fill avg_price
    Column("qty",               Float,      nullable=False),  # total shares requested
    Column("entry_shares",      Integer),                     # = qty (integer copy)
    Column("remaining",         Integer),                     # decrements as TPs fire
    # ORB range (needed for trail computations)
    Column("orb_range",         Float),
    # Stop levels
    Column("stop_price",        Float),                       # original stop (frozen)
    Column("current_stop",      Float),                       # mutable: moves to breakeven
    # TP targets
    Column("tp1_price",         Float),
    Column("tp2_price",         Float),
    # TP share allocations
    Column("tp1_shares",        Integer),
    Column("tp2_shares",        Integer),
    Column("tp3_shares",        Integer),
    # TP state flags
    Column("tp1_hit",           Boolean,    default=False),
    Column("tp2_hit",           Boolean,    default=False),
    Column("tp3_hit",           Boolean,    default=False),
    # Legacy column — kept for backward compat with old state dbs
    Column("tp3_ema_trail",     Boolean,    default=True),
    # Trailing stop after TP1
    Column("use_trail_atp1",    Boolean,    default=False),
    Column("trail_atp1_peak",   Float),
    Column("trail_atp1_dist",   Float,      default=0.0),
    # Running MFE trackers
    Column("max_fav",           Float,      default=0.0),
    Column("post_tp2_mfe",      Float,      default=0.0),
    # Decision / audit
    Column("decision_reason",   String(128)),
    # Exit fields (populated when closed)
    Column("exit_reason",       String(32)),
    Column("exit_price",        Float),
    Column("exit_time",         DateTime),
    # Timestamps / order ref
    Column("opened_at",         DateTime,   nullable=False),
    Column("broker_order_id",   String(64)),
    Column("stop_order_id",     String(64)),
    Column("tp1_order_id",      String(64)),  # OCA bracket TP1 leg order ID
)

closed_trades = Table("closed_trades", _metadata,
    Column("id",                  Integer, primary_key=True, autoincrement=True),
    Column("trade_date",          Date,    nullable=False),
    Column("symbol",              String(16), nullable=False),
    Column("direction",           Integer),
    Column("entry_price",         Float),
    Column("exit_price",          Float),   # target price (backtest-parity field)
    Column("realized_exit_price", Float),   # actual IB fill avg_price
    Column("qty",                 Float),
    Column("pnl_pct",             Float),
    Column("dollar_pnl",          Float),
    Column("exit_reason",         String(32)),  # tp1/tp2/tp3/eod/stop/kill
    Column("opened_at",           DateTime),
    Column("closed_at",           DateTime),
)

equity_curve = Table("equity_curve", _metadata,
    Column("trade_date",   Date,  primary_key=True),
    Column("start_equity", Float, nullable=False),
    Column("end_equity",   Float, nullable=False),
    Column("session_pnl",  Float, nullable=False),
    Column("recorded_at",  DateTime, nullable=False),
)

sigma_calibration = Table("sigma_calibration", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("underlying",   String(16), nullable=False),
    Column("sigma",        Float,      nullable=False),
    Column("lookback_days",Integer),
    Column("n_obs",        Integer),
    Column("calibrated_at",DateTime,   nullable=False),
    Column("source",       String(32)),  # "reference" / "recalibrated"
)

alert_log = Table("alert_log", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("trade_date",   Date),
    Column("level",        String(16), nullable=False),  # INFO/WARN/ERROR/CRITICAL
    Column("category",     String(32)),  # kill_switch / ps_filter / gap_scan / etc.
    Column("symbol",       String(16)),
    Column("message",      Text,       nullable=False),
    Column("created_at",   DateTime,   nullable=False),
)

system_events = Table("system_events", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("event_type",   String(32),  nullable=False),  # startup/shutdown/reconnect
    Column("detail",       Text),
    Column("created_at",   DateTime,    nullable=False),
)

underlying_bars = Table("underlying_bars", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("bar_date",     Date,    nullable=False),
    Column("underlying",   String(16), nullable=False),
    Column("open",         Float),
    Column("high",         Float),
    Column("low",          Float),
    Column("close",        Float,   nullable=False),
    Column("volume",       Float),
    Column("fetched_at",   DateTime, nullable=False),
)

# 15. rtg_history — rolling per-symbol RTG values for percentile computation
rtg_history = Table("rtg_history", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("symbol",       String(16), nullable=False),
    Column("bar_date",     Date,    nullable=False),
    Column("rtg_val",      Float,   nullable=False),
    Column("gap_abs",      Float),
    Column("gap_direction",Integer),
    Column("stored_at",    DateTime, nullable=False),
)

# 16. liquidity_metrics — ADV / DV data per symbol per check date
liquidity_metrics = Table("liquidity_metrics", _metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("check_date",        Date,    nullable=False),
    Column("symbol",            String(16), nullable=False),
    Column("adv_dollars",       Float),
    Column("min_dv_20d",        Float),
    Column("yesterday_dv",      Float),
    Column("intended_dollars",  Float),
    Column("intended_pct_of_adv", Float),
    Column("passed",            Boolean, nullable=False),
    Column("reason",            String(64)),
    Column("warnings",          Text),
    Column("checked_at",        DateTime, nullable=False),
)

# 17. candidates — per-symbol per-session qualification decision log
candidates = Table("candidates", _metadata,
    Column("id",               Integer, primary_key=True, autoincrement=True),
    Column("session_date",     Date,    nullable=False),
    Column("symbol",           String(16), nullable=False),
    Column("phase",            Integer),     # 1 or 2
    Column("gap_abs",          Float),
    Column("gap_direction",    Integer),
    Column("prior_close",      Float),
    Column("ps_filter_passed", Boolean),
    Column("ps_filter_warning",Boolean,  default=False),
    Column("preflight_passed", Boolean),
    Column("preflight_reason", String(128)),
    Column("decision",         String(32)),  # candidate/gap_too_small/dow_excluded/...
    Column("intended_direction",Integer),
    Column("recorded_at",      DateTime, nullable=False),
)

# 18. sigma_history — audit trail of sigma recalibrations
sigma_history = Table("sigma_history", _metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("underlying",   String(16), nullable=False),
    Column("sigma",        Float,   nullable=False),
    Column("lookback_years",Float),
    Column("n_obs",        Integer),
    Column("source",       String(32)),   # "yfinance" / "reference"
    Column("calibrated_at",DateTime, nullable=False),
)


# 19. fills — individual order fill records per leg
fills = Table("fills", _metadata,
    Column("id",             Integer,    primary_key=True, autoincrement=True),
    Column("session_date",   Date,       nullable=False),
    Column("symbol",         String(16), nullable=False),
    Column("side",           String(8),  nullable=False),   # buy/sell
    Column("qty",            Integer,    nullable=False),    # filled shares
    Column("avg_price",      Float,      nullable=False),
    Column("order_id",       String(128)),
    Column("leg",            String(16), nullable=False),   # entry/tp1/tp2/tp3/stop/eod
    Column("attempts",       Integer,    default=1),
    Column("reason",         String(32), nullable=False),   # filled/partial_unfilled/unfilled
    Column("submitted_at",   DateTime,   nullable=False),
    Column("filled_at",      DateTime),
    Column("raw_response_json", Text),
)

# 20. indicator_state — per-bar EMA snapshot for audit / restart recovery
indicator_state = Table("indicator_state", _metadata,
    Column("id",           Integer,    primary_key=True, autoincrement=True),
    Column("session_date", Date,       nullable=False),
    Column("symbol",       String(16), nullable=False),
    Column("ts",           DateTime,   nullable=False),
    Column("ema_value",    Float,      nullable=False),
    Column("stored_at",    DateTime,   nullable=False),
)


# ── Engine factory ────────────────────────────────────────────────────────────

def _enable_wal(dbapi_conn, connection_record):
    """Enable WAL mode for better concurrent read performance."""
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")


def create_db_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    event.listen(engine, "connect", _enable_wal)
    return engine


def migrate_db(engine: Engine) -> None:
    """ALTER TABLE to add any model columns missing from existing DB tables.

    SQLite only supports ADD COLUMN — this handles the common case where a
    column is added to the model after a live.db was already created.  Safe
    to call repeatedly; existing columns are left untouched.
    """
    with engine.begin() as conn:
        for table_name, table in _metadata.tables.items():
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            if not rows:
                continue  # table doesn't exist yet; create_all will handle it
            existing_cols = {row[1].lower() for row in rows}
            for col in table.columns:
                if col.name.lower() in existing_cols:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                conn.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}")
                )


def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist yet, then add any missing columns."""
    _metadata.create_all(engine)
    migrate_db(engine)


# ── StateStore facade ─────────────────────────────────────────────────────────

class StateStore:
    """
    Thin facade over the SQLite database.

    All writes go through this class so that the rest of the system never
    needs to know about table names or SQL syntax.
    """

    def __init__(self, db_path: Path):
        self.engine = create_db_engine(db_path)
        init_db(self.engine)

    def close(self) -> None:
        """Release all pooled connections.  Safe to call more than once."""
        if self.engine is not None:
            self.engine.dispose()

    def conn(self):
        return self.engine.connect()

    # ── day_state ─────────────────────────────────────────────────────────────

    def get_day_state(self, trade_date: date) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute(
                day_state.select().where(day_state.c.trade_date == trade_date)
            ).mappings().first()
            return dict(row) if row else None

    def upsert_day_state(self, trade_date: date, **kwargs) -> None:
        now = datetime.now(UTC)
        existing = self.get_day_state(trade_date)
        with self.conn() as c:
            if existing is None:
                c.execute(day_state.insert().values(
                    trade_date=trade_date, created_at=now, updated_at=now, **kwargs
                ))
            else:
                c.execute(
                    day_state.update()
                    .where(day_state.c.trade_date == trade_date)
                    .values(updated_at=now, **kwargs)
                )
            c.commit()

    def set_kill_switch(self, trade_date: date, reason: str) -> None:
        self.upsert_day_state(trade_date, kill_triggered=True, kill_reason=reason)

    def is_kill_triggered(self, trade_date: date) -> bool:
        row = self.get_day_state(trade_date)
        return bool(row and row.get("kill_triggered"))

    # ── gap_scan ──────────────────────────────────────────────────────────────

    def save_gap_scan(self, trade_date: date, symbol: str, **kwargs) -> None:
        with self.conn() as c:
            c.execute(gap_scan.insert().values(
                trade_date=trade_date, symbol=symbol,
                scanned_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    def get_qualifying_symbols(self, trade_date: date) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                gap_scan.select()
                .where(gap_scan.c.trade_date == trade_date)
                .where(gap_scan.c.qualifies == True)  # noqa: E712
            ).mappings().all()
            return [r["symbol"] for r in rows]

    # ── ps_filter_result ──────────────────────────────────────────────────────

    def save_ps_filter(self, trade_date: date, symbol: str, **kwargs) -> None:
        with self.conn() as c:
            c.execute(ps_filter_result.insert().values(
                trade_date=trade_date, symbol=symbol,
                checked_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    def get_ps_passed_symbols(self, trade_date: date) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                ps_filter_result.select()
                .where(ps_filter_result.c.trade_date == trade_date)
                .where(ps_filter_result.c.passed == True)  # noqa: E712
            ).mappings().all()
            return [r["symbol"] for r in rows]

    # ── breakout_signal ───────────────────────────────────────────────────────

    def save_breakout_signal(
        self,
        trade_date: date,
        symbol: str,
        direction: int,
        breakout_price: float,
        orb_high: float,
        orb_low: float,
    ) -> None:
        with self.conn() as c:
            c.execute(breakout_signal.insert().values(
                trade_date=trade_date,
                symbol=symbol,
                direction=direction,
                breakout_price=breakout_price,
                orb_high=orb_high,
                orb_low=orb_low,
                detected_at=datetime.now(UTC),
            ))
            c.commit()

    # ── open_positions ────────────────────────────────────────────────────────

    def save_open_position(self, symbol: str, trade_date: date, **kwargs) -> None:
        with self.conn() as c:
            c.execute(open_positions.insert().values(
                symbol=symbol, trade_date=trade_date,
                opened_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    def update_open_position(self, symbol: str, **kwargs) -> None:
        """Update mutable fields of an open position (stop, TP flags, etc.)."""
        with self.conn() as c:
            c.execute(
                open_positions.update()
                .where(open_positions.c.symbol == symbol)
                .values(**kwargs)
            )
            c.commit()

    def close_position(self, symbol: str) -> None:
        with self.conn() as c:
            c.execute(open_positions.delete().where(open_positions.c.symbol == symbol))
            c.commit()

    def get_open_position(self, symbol: str) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute(
                open_positions.select().where(open_positions.c.symbol == symbol)
            ).mappings().first()
            return dict(row) if row else None

    def all_open_positions(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(open_positions.select()).mappings().all()
            return [dict(r) for r in rows]

    def all_open_symbols(self) -> list[str]:
        with self.conn() as c:
            rows = c.execute(open_positions.select()).mappings().all()
            return [r["symbol"] for r in rows]

    # ── closed_trades ─────────────────────────────────────────────────────────

    def save_closed_trade(self, **kwargs) -> None:
        with self.conn() as c:
            c.execute(closed_trades.insert().values(
                closed_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    # ── equity_curve ──────────────────────────────────────────────────────────

    def record_equity(self, trade_date: date, start_equity: float,
                      end_equity: float, session_pnl: float) -> None:
        with self.conn() as c:
            c.execute(equity_curve.insert().values(
                trade_date=trade_date,
                start_equity=start_equity,
                end_equity=end_equity,
                session_pnl=session_pnl,
                recorded_at=datetime.now(UTC),
            ))
            c.commit()

    # ── alert_log ─────────────────────────────────────────────────────────────

    def log_alert(self, level: str, message: str, category: str = "",
                  symbol: str = "", trade_date: Optional[date] = None) -> None:
        with self.conn() as c:
            c.execute(alert_log.insert().values(
                trade_date=trade_date,
                level=level,
                category=category,
                symbol=symbol,
                message=message,
                created_at=datetime.now(UTC),
            ))
            c.commit()

    # ── system_events ─────────────────────────────────────────────────────────

    def log_event(self, event_type: str, detail: str = "") -> None:
        with self.conn() as c:
            c.execute(system_events.insert().values(
                event_type=event_type,
                detail=detail,
                created_at=datetime.now(UTC),
            ))
            c.commit()

    # ── underlying_bars ───────────────────────────────────────────────────────

    def save_underlying_bar(self, bar_date: date, underlying: str, **kwargs) -> None:
        with self.conn() as c:
            c.execute(underlying_bars.insert().values(
                bar_date=bar_date, underlying=underlying,
                fetched_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    def get_latest_underlying_close(self, underlying: str) -> Optional[float]:
        with self.conn() as c:
            row = c.execute(
                underlying_bars.select()
                .where(underlying_bars.c.underlying == underlying)
                .order_by(underlying_bars.c.bar_date.desc())
                .limit(1)
            ).mappings().first()
            return row["close"] if row else None

    # ── rtg_history ───────────────────────────────────────────────────────────

    def save_rtg(self, symbol: str, bar_date: date, rtg_val: float,
                 gap_abs: float = None, gap_direction: int = None) -> None:
        with self.conn() as c:
            c.execute(rtg_history.insert().values(
                symbol=symbol, bar_date=bar_date, rtg_val=rtg_val,
                gap_abs=gap_abs, gap_direction=gap_direction,
                stored_at=datetime.now(UTC),
            ))
            c.commit()

    def get_rtg_history(self, symbol: str, before_date: date,
                        window: int = 252) -> list[float]:
        with self.conn() as c:
            rows = c.execute(
                rtg_history.select()
                .where(rtg_history.c.symbol == symbol)
                .where(rtg_history.c.bar_date < before_date)
                .order_by(rtg_history.c.bar_date.desc())
                .limit(window)
            ).mappings().all()
            return [r["rtg_val"] for r in reversed(rows)]

    # ── liquidity_metrics ─────────────────────────────────────────────────────

    def save_liquidity_metrics(self, check_date: date, symbol: str,
                               **kwargs) -> None:
        with self.conn() as c:
            c.execute(liquidity_metrics.insert().values(
                check_date=check_date, symbol=symbol,
                checked_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    # ── candidates ────────────────────────────────────────────────────────────

    def save_candidate(self, session_date: date, symbol: str,
                       **kwargs) -> None:
        with self.conn() as c:
            c.execute(candidates.insert().values(
                session_date=session_date, symbol=symbol,
                recorded_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    def get_candidates(self, session_date: date,
                       phase: int = None) -> list[dict]:
        with self.conn() as c:
            q = candidates.select().where(
                candidates.c.session_date == session_date
            )
            if phase is not None:
                q = q.where(candidates.c.phase == phase)
            rows = c.execute(q).mappings().all()
            return [dict(r) for r in rows]

    def get_final_candidates(self, session_date: date) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                candidates.select()
                .where(candidates.c.session_date == session_date)
                .where(candidates.c.decision == "candidate")
            ).mappings().all()
            return [r["symbol"] for r in rows]

    # ── sigma_history ─────────────────────────────────────────────────────────

    def save_sigma_calibration(self, underlying: str, sigma: float,
                               **kwargs) -> None:
        with self.conn() as c:
            c.execute(sigma_history.insert().values(
                underlying=underlying, sigma=sigma,
                calibrated_at=datetime.now(UTC), **kwargs,
            ))
            c.commit()

    # ── fills ─────────────────────────────────────────────────────────────────

    def save_fill(self, session_date: date, symbol: str, **kwargs) -> None:
        with self.conn() as c:
            c.execute(fills.insert().values(
                session_date=session_date, symbol=symbol, **kwargs,
            ))
            c.commit()

    def get_fills(self, session_date: date, symbol: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                fills.select()
                .where(fills.c.session_date == session_date)
                .where(fills.c.symbol == symbol)
                .order_by(fills.c.id)
            ).mappings().all()
            return [dict(r) for r in rows]

    # ── indicator_state ───────────────────────────────────────────────────────

    def save_indicator_state(self, session_date: date, symbol: str,
                             ts: datetime, ema_value: float) -> None:
        with self.conn() as c:
            c.execute(indicator_state.insert().values(
                session_date=session_date, symbol=symbol, ts=ts,
                ema_value=ema_value, stored_at=datetime.now(UTC),
            ))
            c.commit()

    def get_latest_indicator(self, session_date: date,
                             symbol: str) -> Optional[float]:
        with self.conn() as c:
            row = c.execute(
                indicator_state.select()
                .where(indicator_state.c.session_date == session_date)
                .where(indicator_state.c.symbol == symbol)
                .order_by(indicator_state.c.ts.desc())
                .limit(1)
            ).mappings().first()
            return float(row["ema_value"]) if row else None
