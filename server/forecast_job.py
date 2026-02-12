import bisect
import json
import math
import os
import traceback
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pymysql
import pytz
import requests
import xgboost as xgb

TZ_NAME = "America/Chicago"
TZ = pytz.timezone(TZ_NAME)

RESAMPLE_MINUTES = int(os.getenv("RESAMPLE_MINUTES", "15"))
MIN_SAMPLES_PER_LOC = int(os.getenv("MIN_SAMPLES_PER_LOC", "80"))
MIN_SAMPLES_PER_HOUR_TOTAL = int(os.getenv("MIN_SAMPLES_PER_HOUR_TOTAL", "5"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "300"))
TRAIN_SPLIT = float(os.getenv("MODEL_TRAIN_SPLIT", "0.8"))
EARLY_STOPPING_ROUNDS = int(os.getenv("MODEL_EARLY_STOPPING", "50"))
MODEL_MAX_DEPTH = int(os.getenv("MODEL_MAX_DEPTH", "4"))
MODEL_NUM_BOOST_ROUND = int(os.getenv("MODEL_NUM_BOOST_ROUND", "500"))
MODEL_NTHREAD = int(os.getenv("MODEL_NTHREAD", "2"))
MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
MODEL_GUARDRAIL_MAX_MAE_DEGRADE = float(os.getenv("MODEL_GUARDRAIL_MAX_MAE_DEGRADE", "0.05"))
MODEL_GUARDRAIL_MIN_VAL_ROWS = int(os.getenv("MODEL_GUARDRAIL_MIN_VAL_ROWS", "200"))
FORCE_RETRAIN = os.getenv("MODEL_FORCE_RETRAIN", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_SCHEMA_VERSION = 4

INTERVAL_MIN_SAMPLES_PER_HOUR = int(os.getenv("INTERVAL_MIN_SAMPLES_PER_HOUR", "30"))
INTERVAL_Q_LOW = float(os.getenv("INTERVAL_Q_LOW", "0.10"))
INTERVAL_Q_HIGH = float(os.getenv("INTERVAL_Q_HIGH", "0.90"))

HISTORY_DAYS = int(os.getenv("GYM_MODEL_HISTORY_DAYS", "0"))
DB_TIMEZONE_NAME = os.getenv("GYM_DB_TIMEZONE", TZ_NAME)
DB_TZ = pytz.timezone(DB_TIMEZONE_NAME)

FORECAST_DAY_START_HOUR = int(os.getenv("FORECAST_DAY_START_HOUR", "6"))
FORECAST_DAY_END_HOUR = int(os.getenv("FORECAST_DAY_END_HOUR", "23"))

PEAK_WINDOW_HOURS = int(os.getenv("PEAK_WINDOW_HOURS", "3"))
PEAK_Z_THRESHOLD = float(os.getenv("PEAK_Z_THRESHOLD", "0.6"))
LOW_Z_THRESHOLD = float(os.getenv("LOW_Z_THRESHOLD", "0.6"))
SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "3"))

STALE_SENSOR_HOURS = float(os.getenv("STALE_SENSOR_HOURS", "24"))
IMPOSSIBLE_JUMP_PCT = float(os.getenv("IMPOSSIBLE_JUMP_PCT", "0.60"))
IMPOSSIBLE_JUMP_MAX_GAP_MIN = float(os.getenv("IMPOSSIBLE_JUMP_MAX_GAP_MIN", "120"))

SPIKE_AWARE_ENABLED = os.getenv("SPIKE_AWARE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SPIKE_AWARE_MAX_AGE_MIN = float(os.getenv("SPIKE_AWARE_MAX_AGE_MIN", "90"))
SPIKE_AWARE_HORIZON_HOURS = int(os.getenv("SPIKE_AWARE_HORIZON_HOURS", "6"))
SPIKE_AWARE_DECAY = float(os.getenv("SPIKE_AWARE_DECAY", "0.55"))
SPIKE_AWARE_MAX_CAP_MULTIPLIER = float(os.getenv("SPIKE_AWARE_MAX_CAP_MULTIPLIER", "1.35"))
WEATHER_URL = os.getenv("GYM_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
WEATHER_ARCHIVE_URL = os.getenv(
    "GYM_WEATHER_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)
WEATHER_LAT = float(os.getenv("GYM_WEATHER_LAT", "43.0731"))
WEATHER_LON = float(os.getenv("GYM_WEATHER_LON", "-89.4012"))
WEATHER_FORECAST_DAYS = int(os.getenv("GYM_WEATHER_FORECAST_DAYS", "7"))
WEATHER_HISTORY_MAX_DAYS = int(os.getenv("GYM_WEATHER_HISTORY_MAX_DAYS", "180"))

FORECAST_JSON_PATH = os.getenv(
    "FORECAST_JSON_PATH",
    os.path.join(os.path.dirname(__file__), "forecast.json"),
)

MODEL_ARTIFACT_DIR = os.getenv(
    "MODEL_ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "model_artifacts"),
)
MODEL_PATH = os.path.join(MODEL_ARTIFACT_DIR, "forecast_model.xgb.json")
MODEL_META_PATH = os.path.join(MODEL_ARTIFACT_DIR, "forecast_model.meta.json")

WEATHER_KEYS = (
    "temp_c",
    "feels_like_c",
    "precip_mm",
    "rain_mm",
    "snow_cm",
    "wind_mps",
    "wind_gust_mps",
    "humidity_pct",
    "weather_code",
)
WEATHER_ROLLING_KEYS = (
    "temp_c",
    "precip_mm",
    "wind_mps",
)
WEATHER_API_HOURLY_MAP = {
    "temp_c": "temperature_2m",
    "feels_like_c": "apparent_temperature",
    "precip_mm": "precipitation",
    "rain_mm": "rain",
    "snow_cm": "snowfall",
    "wind_mps": "wind_speed_10m",
    "wind_gust_mps": "wind_gusts_10m",
    "humidity_pct": "relative_humidity_2m",
    "weather_code": "weather_code",
}

FACILITIES = {
    1186: {
        "name": "Nicholas Recreation Center",
        "categories": [
            {
                "key": "fitness_floors",
                "title": "Fitness Floors",
                "location_ids": [5761, 5760, 5762, 5758],
            },
            {
                "key": "basketball_courts",
                "title": "Basketball Courts",
                "location_ids": [7089, 7090, 5766],
            },
            {
                "key": "running_track",
                "title": "Running Track",
                "location_ids": [5763],
            },
            {
                "key": "swimming_pool",
                "title": "Swimming Pool",
                "location_ids": [5764],
            },
            {
                "key": "racquetball_courts",
                "title": "Racquetball Courts",
                "location_ids": [5753, 5754],
            },
        ],
    },
    1656: {
        "name": "Bakke Recreation & Wellbeing Center",
        "categories": [
            {
                "key": "fitness_floors",
                "title": "Fitness Floors",
                "location_ids": [8718, 8717, 8705, 8700, 8699, 8696],
            },
            {
                "key": "basketball_courts",
                "title": "Basketball Courts",
                "location_ids": [8720, 8714, 8698],
            },
            {
                "key": "running_track",
                "title": "Running Track",
                "location_ids": [8694],
            },
            {
                "key": "swimming_pool",
                "title": "Swimming Pool",
                "location_ids": [8716],
            },
            {
                "key": "rock_climbing",
                "title": "Rock Climbing",
                "location_ids": [8701],
            },
            {
                "key": "ice_skating",
                "title": "Ice Skating",
                "location_ids": [10550],
            },
            {
                "key": "esports_room",
                "title": "Esports Room",
                "location_ids": [8712],
            },
            {
                "key": "sports_simulators",
                "title": "Sports Simulators",
                "location_ids": [8695],
            },
        ],
    },
}

SQL_HISTORY_BASE = """
SELECT
    location_id,
    last_updated,
    fetched_at,
    current_capacity,
    is_closed,
    max_capacity
FROM location_history
WHERE current_capacity IS NOT NULL
  AND (is_closed = 0 OR is_closed IS NULL)
"""


def db_connect():
    host = os.getenv("GYM_DB_HOST", "localhost")
    port = int(os.getenv("GYM_DB_PORT", "3306"))
    user = os.getenv("GYM_DB_USER", "root")
    password = os.getenv("GYM_DB_PASSWORD", "faDpud-vydqys-batto9")
    database = os.getenv("GYM_DB_NAME", "gym_data")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
        charset="utf8mb4",
    )


def all_location_ids() -> List[int]:
    ids = []
    for facility in FACILITIES.values():
        for category in facility["categories"]:
            ids.extend(category["location_ids"])
    return sorted(set(ids))


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = DB_TZ.localize(dt)
    else:
        dt = dt.astimezone(DB_TZ)
    return dt.astimezone(TZ)


def floor_time(dt: datetime, minutes: int) -> datetime:
    minutes = max(1, minutes)
    dt = dt.replace(second=0, microsecond=0)
    return dt.replace(minute=dt.minute - (dt.minute % minutes))


def aggregate_sum_count(store: Dict, key, value: float) -> None:
    if key in store:
        store[key][0] += value
        store[key][1] += 1.0
    else:
        store[key] = [value, 1.0]


def finalize_averages(store: Dict) -> Dict:
    return {k: (v[0] / v[1], int(v[1])) for k, v in store.items()}


def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        return TZ.localize(dt)
    return dt.astimezone(TZ)


def parse_observed_at_value(raw_value) -> Optional[datetime]:
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        return to_local(raw_value)

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None

        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = None
        try:
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(normalized, fmt)
                    break
                except Exception:
                    continue

        if parsed is None:
            return None

        if parsed.tzinfo is None:
            parsed = DB_TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(DB_TZ)
        return parsed.astimezone(TZ)

    return None


def model_feature_count(loc_count: int) -> int:
    # 13 time/cycle + 9 lag/recency + (9 weather * 3 stats) + (3 rolling weather) + loc one-hot
    return 52 + loc_count


def last_before(times: List[datetime], values: List[float], target: datetime) -> Tuple[float, float]:
    idx = bisect.bisect_left(times, target) - 1
    if idx >= 0:
        return values[idx], (target - times[idx]).total_seconds() / 60.0
    return float("nan"), float("nan")


def build_time_features(dt: datetime) -> List[float]:
    hour = dt.hour
    dow = dt.weekday()
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    is_weekend = 1 if dow >= 5 else 0

    hour_rad = 2 * math.pi * hour / 24
    dow_rad = 2 * math.pi * dow / 7
    month_rad = 2 * math.pi * (month - 1) / 12
    doy_rad = 2 * math.pi * day_of_year / 365.25

    return [
        float(hour),
        float(dow),
        float(month),
        float(day_of_year),
        float(is_weekend),
        math.sin(hour_rad),
        math.cos(hour_rad),
        math.sin(dow_rad),
        math.cos(dow_rad),
        math.sin(month_rad),
        math.cos(month_rad),
        math.sin(doy_rad),
        math.cos(doy_rad),
    ]


def rolling_mean(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
) -> float:
    values = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
        value = bucket_map.get(ts)
        if value is not None:
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def weather_last_before(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    key: str,
) -> float:
    idx = bisect.bisect_left(times, target) - 1
    while idx >= 0:
        row = weather_map.get(times[idx], {})
        value = row.get(key)
        if value is not None:
            return float(value)
        idx -= 1
    return float("nan")


def weather_value_at_or_before(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    key: str,
) -> float:
    row = weather_map.get(target)
    if row is not None:
        value = row.get(key)
        if value is not None:
            return float(value)
    return weather_last_before(times, weather_map, target, key)


def weather_rolling_mean(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    steps: int,
    key: str,
) -> float:
    values = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
        value = weather_value_at_or_before(times, weather_map, ts, key)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_features(
    target: datetime,
    loc_data: Dict[str, object],
    onehot: List[float],
    weather_source: Optional[Dict[str, object]] = None,
) -> List[float]:
    bucket_map = loc_data["bucket_map"]
    bucket_times = loc_data["bucket_times"]
    bucket_values = loc_data["bucket_values"]
    raw_times = loc_data["raw_times"]
    raw_values = loc_data["raw_values"]
    if weather_source is not None:
        weather_bucket_map = weather_source.get("map", {})
        weather_bucket_times = weather_source.get("times", [])
    else:
        weather_bucket_map = loc_data.get("weather_bucket_map", {})
        weather_bucket_times = loc_data.get("weather_bucket_times", [])

    lag_15m = bucket_map.get(target - timedelta(minutes=RESAMPLE_MINUTES))
    lag_1h = bucket_map.get(target - timedelta(hours=1))
    lag_2h = bucket_map.get(target - timedelta(hours=2))

    if lag_15m is None:
        lag_15m = float("nan")
    if lag_1h is None:
        lag_1h = float("nan")
    if lag_2h is None:
        lag_2h = float("nan")

    last_bucket, minutes_since_bucket = last_before(bucket_times, bucket_values, target)
    _last_raw, minutes_since_raw = last_before(raw_times, raw_values, target)

    delta_1h = float("nan")
    if not math.isnan(lag_15m) and not math.isnan(lag_1h):
        delta_1h = lag_15m - lag_1h

    roll_1h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(60 / RESAMPLE_MINUTES)),
    )
    roll_2h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(120 / RESAMPLE_MINUTES)),
    )
    weather_features: List[float] = []
    weather_ts = target
    lag_ts = target - timedelta(hours=1)

    for key in WEATHER_KEYS:
        weather_now = weather_value_at_or_before(
            weather_bucket_times,
            weather_bucket_map,
            weather_ts,
            key,
        )
        weather_1h = weather_value_at_or_before(
            weather_bucket_times,
            weather_bucket_map,
            lag_ts,
            key,
        )
        weather_delta_1h = float("nan")
        if not math.isnan(weather_now) and not math.isnan(weather_1h):
            weather_delta_1h = weather_now - weather_1h

        weather_features.extend(
            [
                float(weather_now),
                float(weather_1h),
                float(weather_delta_1h),
            ]
        )

    for key in WEATHER_ROLLING_KEYS:
        weather_roll = weather_rolling_mean(
            weather_bucket_times,
            weather_bucket_map,
            target,
            steps=max(1, int(180 / RESAMPLE_MINUTES)),
            key=key,
        )
        weather_features.append(float(weather_roll))

    return (
        build_time_features(target)
        + [
            float(lag_15m),
            float(lag_1h),
            float(lag_2h),
            float(delta_1h),
            float(roll_1h),
            float(roll_2h),
            float(last_bucket),
            float(minutes_since_bucket),
            float(minutes_since_raw),
        ]
        + weather_features
        + onehot
    )


def dedupe_exact_timestamps(entries: List[Tuple[datetime, float]]) -> Tuple[List[Tuple[datetime, float]], int]:
    if not entries:
        return [], 0
    entries.sort(key=lambda row: row[0])
    deduped = []
    removed = 0
    idx = 0
    n = len(entries)
    while idx < n:
        ts = entries[idx][0]
        values = [entries[idx][1]]
        idx += 1
        while idx < n and entries[idx][0] == ts:
            values.append(entries[idx][1])
            idx += 1
        if len(values) > 1:
            removed += len(values) - 1
        deduped.append((ts, float(sum(values) / len(values))))
    return deduped, removed


def drop_impossible_jumps(
    entries: List[Tuple[datetime, float]],
    max_cap: int,
) -> Tuple[List[Tuple[datetime, float]], int]:
    if not entries:
        return [], 0

    cleaned = [entries[0]]
    removed = 0
    max_jump = max_cap * IMPOSSIBLE_JUMP_PCT

    for ts, value in entries[1:]:
        prev_ts, prev_value = cleaned[-1]
        gap_min = (ts - prev_ts).total_seconds() / 60.0
        jump = abs(value - prev_value)
        if gap_min <= IMPOSSIBLE_JUMP_MAX_GAP_MIN and jump > max_jump:
            removed += 1
            continue
        cleaned.append((ts, value))

    return cleaned, removed


def load_history(conn):
    raw_by_loc: Dict[int, List[Tuple[datetime, float]]] = {}
    max_caps: Dict[int, int] = {}

    quality = {
        "rowsRead": 0,
        "rowsDroppedInvalid": 0,
        "duplicatesRemoved": 0,
        "impossibleJumpsRemoved": 0,
        "staleLocations": [],
    }

    sql = SQL_HISTORY_BASE
    params: Tuple[object, ...] = ()
    if HISTORY_DAYS > 0:
        sql += " AND fetched_at >= %s"
        since = datetime.now(DB_TZ) - timedelta(days=HISTORY_DAYS)
        params = (since,)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        for (
            loc_id,
            last_updated,
            fetched_at,
            current_capacity,
            _is_closed,
            max_cap,
        ) in cur.fetchall():
            quality["rowsRead"] += 1
            if current_capacity is None:
                quality["rowsDroppedInvalid"] += 1
                continue

            try:
                loc_id = int(loc_id)
                current_capacity = float(current_capacity)
            except Exception:
                quality["rowsDroppedInvalid"] += 1
                continue

            local_dt = parse_observed_at_value(last_updated)
            if local_dt is None and fetched_at is not None:
                local_dt = to_local(fetched_at)
            if local_dt is None:
                quality["rowsDroppedInvalid"] += 1
                continue

            raw_by_loc.setdefault(loc_id, []).append((local_dt, current_capacity))

            if max_cap is not None:
                try:
                    cap = int(max_cap)
                    if cap > 0:
                        max_caps[loc_id] = max(cap, max_caps.get(loc_id, 0))
                except Exception:
                    pass

    avg_dow_hour_sum: Dict[Tuple[int, int, int], List[float]] = {}
    avg_hour_sum: Dict[Tuple[int, int], List[float]] = {}
    avg_overall_sum: Dict[int, List[float]] = {}

    loc_data: Dict[int, Dict[str, object]] = {}
    loc_samples: Dict[int, int] = {}
    now_local = datetime.now(TZ)

    for loc_id, entries in raw_by_loc.items():
        max_cap = max_caps.get(loc_id, 0)
        if max_cap <= 0:
            continue

        deduped, dup_removed = dedupe_exact_timestamps(entries)
        quality["duplicatesRemoved"] += dup_removed

        cleaned, jump_removed = drop_impossible_jumps(deduped, max_cap=max_cap)
        quality["impossibleJumpsRemoved"] += jump_removed

        if not cleaned:
            continue

        latest_ts = cleaned[-1][0]
        age_hours = (now_local - latest_ts).total_seconds() / 3600.0
        is_stale = age_hours > STALE_SENSOR_HOURS
        if is_stale:
            quality["staleLocations"].append(loc_id)

        raw_times = [row[0] for row in cleaned]
        raw_values = [float(row[1]) for row in cleaned]

        bucket_counts: Dict[datetime, List[float]] = {}
        for ts, count in cleaned:
            bucket = floor_time(ts, RESAMPLE_MINUTES)
            bucket_counts.setdefault(bucket, []).append(count)

        if not bucket_counts:
            continue

        bucket_times = sorted(bucket_counts.keys())
        bucket_values: List[float] = []
        bucket_map: Dict[datetime, float] = {}

        for bt in bucket_times:
            avg_count = float(sum(bucket_counts[bt]) / len(bucket_counts[bt]))
            ratio = avg_count / max_cap
            ratio = max(0.0, min(ratio, 1.2))
            bucket_values.append(ratio)
            bucket_map[bt] = ratio

            aggregate_sum_count(avg_dow_hour_sum, (loc_id, bt.weekday(), bt.hour), ratio)
            aggregate_sum_count(avg_hour_sum, (loc_id, bt.hour), ratio)
            aggregate_sum_count(avg_overall_sum, loc_id, ratio)

        loc_data[loc_id] = {
            "raw_times": raw_times,
            "raw_values": raw_values,
            "bucket_times": bucket_times,
            "bucket_values": bucket_values,
            "bucket_map": bucket_map,
            "max_cap": max_cap,
            "is_stale": is_stale,
            "latest_ts": latest_ts.isoformat(),
        }
        loc_samples[loc_id] = len(bucket_times)

    quality["locationsWithHistory"] = len(loc_data)
    quality["staleLocations"] = sorted(set(quality["staleLocations"]))
    quality["staleLocationsCount"] = len(quality["staleLocations"])

    avg_dow_hour = finalize_averages(avg_dow_hour_sum)
    avg_hour = finalize_averages(avg_hour_sum)
    avg_overall = finalize_averages(avg_overall_sum)

    return loc_data, avg_dow_hour, avg_hour, avg_overall, max_caps, loc_samples, quality


def build_onehot(loc_ids: Iterable[int]) -> Dict[int, List[float]]:
    unique = sorted(set(loc_ids))
    vectors: Dict[int, List[float]] = {}
    for idx, loc_id in enumerate(unique):
        vec = [0.0] * len(unique)
        vec[idx] = 1.0
        vectors[loc_id] = vec
    return vectors


def build_interval_profile(
    val_times: List[datetime],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    residuals = y_true - y_pred
    profile: Dict[str, object] = {
        "global": {
            "q10": float(np.quantile(residuals, INTERVAL_Q_LOW)),
            "q90": float(np.quantile(residuals, INTERVAL_Q_HIGH)),
            "count": int(len(residuals)),
        },
        "byHour": {},
    }

    for hour in range(24):
        idx = [i for i, ts in enumerate(val_times) if ts.hour == hour]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        hour_res = residuals[idx]
        profile["byHour"][str(hour)] = {
            "q10": float(np.quantile(hour_res, INTERVAL_Q_LOW)),
            "q90": float(np.quantile(hour_res, INTERVAL_Q_HIGH)),
            "count": int(len(hour_res)),
        }

    return profile


def train_global_model(
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    loc_samples: Dict[int, int],
    weather_source: Optional[Dict[str, object]],
):
    rows = []
    for loc_id, data in loc_data.items():
        if loc_samples.get(loc_id, 0) < MIN_SAMPLES_PER_LOC:
            continue
        if data.get("is_stale"):
            continue
        onehot_vec = onehot.get(loc_id)
        if onehot_vec is None:
            continue
        for target, label in zip(data["bucket_times"], data["bucket_values"]):
            features = build_features(target, data, onehot_vec, weather_source=weather_source)
            rows.append((target, features, label))

    if len(rows) < MIN_TRAIN_SAMPLES:
        return None, {
            "train_rows": len(rows),
            "val_rows": 0,
            "val_mae": None,
            "val_rmse": None,
        }, None

    times = [row[0] for row in rows]
    X = np.array([row[1] for row in rows], dtype=np.float32)
    y = np.array([row[2] for row in rows], dtype=np.float32)

    times_sorted = sorted(times)
    split_idx = int(len(times_sorted) * TRAIN_SPLIT)
    split_idx = max(1, min(split_idx, len(times_sorted) - 1))
    split_time = times_sorted[split_idx]

    train_idx = [i for i, ts in enumerate(times) if ts < split_time]
    val_idx = [i for i, ts in enumerate(times) if ts >= split_time]

    if len(train_idx) < 10 or len(val_idx) < 10:
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": MODEL_MAX_DEPTH,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "lambda": 1.0,
                "alpha": 0.0,
                "verbosity": 0,
                "nthread": MODEL_NTHREAD,
            },
            dtrain,
            num_boost_round=MODEL_NUM_BOOST_ROUND,
        )
        return model, {
            "train_rows": len(rows),
            "val_rows": 0,
            "val_mae": None,
            "val_rmse": None,
        }, None

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    val_times = [times[i] for i in val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "max_depth": MODEL_MAX_DEPTH,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0,
        "verbosity": 0,
        "nthread": MODEL_NTHREAD,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=MODEL_NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    mae = float(np.mean(np.abs(preds - y_val)))
    rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
    interval_profile = build_interval_profile(val_times, y_val, preds)

    return model, {
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_mae": mae,
        "val_rmse": rmse,
    }, interval_profile


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_saved_model(expected_loc_ids: List[int], expected_feature_count: int):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_META_PATH):
        return None, None

    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None, None

    if meta.get("schemaVersion") != MODEL_SCHEMA_VERSION:
        return None, None
    if meta.get("locIds") != expected_loc_ids:
        return None, None
    if int(meta.get("featureCount", -1)) != expected_feature_count:
        return None, None

    model = xgb.Booster()
    try:
        model.load_model(MODEL_PATH)
    except Exception:
        return None, None

    return model, meta


def save_model_artifacts(model: xgb.Booster, meta: Dict[str, object]) -> None:
    ensure_dir(MODEL_ARTIFACT_DIR)
    tmp_model = MODEL_PATH + ".tmp"
    tmp_meta = MODEL_META_PATH + ".tmp"

    model.save_model(tmp_model)
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)

    os.replace(tmp_model, MODEL_PATH)
    os.replace(tmp_meta, MODEL_META_PATH)


def should_retrain_model(meta: Optional[Dict[str, object]], now: datetime) -> bool:
    if FORCE_RETRAIN:
        return True
    if not meta:
        return True

    trained_at = parse_iso_datetime(meta.get("trainedAt", ""))
    if not trained_at:
        return True

    return (now - trained_at) >= timedelta(hours=MODEL_RETRAIN_HOURS)


def passes_guardrail(
    baseline_meta: Optional[Dict[str, object]],
    candidate_metrics: Dict[str, object],
) -> bool:
    if not baseline_meta:
        return True

    baseline_mae = baseline_meta.get("valMae")
    baseline_rows = int(baseline_meta.get("valRows", 0) or 0)
    candidate_mae = candidate_metrics.get("val_mae")
    candidate_rows = int(candidate_metrics.get("val_rows", 0) or 0)

    if baseline_mae is None or candidate_mae is None:
        return True
    if baseline_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True
    if candidate_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True

    return float(candidate_mae) <= float(baseline_mae) * (1.0 + MODEL_GUARDRAIL_MAX_MAE_DEGRADE)


def prepare_model(
    now: datetime,
    expected_loc_ids: List[int],
    onehot: Dict[int, List[float]],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
):
    feature_count = model_feature_count(len(expected_loc_ids))
    saved_model, saved_meta = load_saved_model(expected_loc_ids, feature_count)

    status = "using_saved_model" if saved_model is not None else "no_saved_model"
    run_metrics: Dict[str, object] = {}

    if should_retrain_model(saved_meta, now):
        candidate_model, candidate_metrics, interval_profile = train_global_model(
            loc_data,
            onehot,
            loc_samples,
            weather_series,
        )
        run_metrics = candidate_metrics

        if candidate_model is None:
            if saved_model is not None:
                status = "using_saved_model_train_skipped"
                return saved_model, saved_meta, status, run_metrics
            status = "no_model_train_skipped"
            return None, None, status, run_metrics

        candidate_meta = {
            "schemaVersion": MODEL_SCHEMA_VERSION,
            "trainedAt": now.isoformat(),
            "locIds": expected_loc_ids,
            "featureCount": feature_count,
            "trainRows": int(candidate_metrics.get("train_rows", 0)),
            "valRows": int(candidate_metrics.get("val_rows", 0)),
            "valMae": candidate_metrics.get("val_mae"),
            "valRmse": candidate_metrics.get("val_rmse"),
            "residualProfile": interval_profile,
        }

        if saved_model is not None and not passes_guardrail(saved_meta, candidate_metrics):
            status = "guardrail_kept_previous"
            return saved_model, saved_meta, status, run_metrics

        save_model_artifacts(candidate_model, candidate_meta)
        status = "trained_and_saved"
        return candidate_model, candidate_meta, status, run_metrics

    return saved_model, saved_meta, status, run_metrics


def interval_bounds(
    point_ratio: float,
    hour: int,
    residual_profile: Optional[Dict[str, object]],
) -> Tuple[float, float, float]:
    point_ratio = max(0.0, min(point_ratio, 1.0))
    if not residual_profile:
        return point_ratio, point_ratio, point_ratio

    hour_stats = residual_profile.get("byHour", {}).get(str(hour))
    if hour_stats and int(hour_stats.get("count", 0)) >= INTERVAL_MIN_SAMPLES_PER_HOUR:
        q10 = float(hour_stats.get("q10", 0.0))
        q90 = float(hour_stats.get("q90", 0.0))
    else:
        global_stats = residual_profile.get("global", {})
        q10 = float(global_stats.get("q10", 0.0))
        q90 = float(global_stats.get("q90", 0.0))

    p10 = max(0.0, min(point_ratio + q10, 1.0))
    p90 = max(0.0, min(point_ratio + q90, 1.0))
    if p10 > p90:
        p10, p90 = p90, p10
    return p10, point_ratio, p90


def get_sample_count(
    loc_id: int,
    target: datetime,
    avg_dow_hour: Dict[Tuple[int, int, int], Tuple[float, int]],
    avg_hour: Dict[Tuple[int, int], Tuple[float, int]],
    avg_overall: Dict[int, Tuple[float, int]],
) -> int:
    dow = target.weekday()
    hour = target.hour

    value = avg_dow_hour.get((loc_id, dow, hour))
    if value:
        return int(value[1])

    value = avg_hour.get((loc_id, hour))
    if value:
        return int(value[1])

    value = avg_overall.get(loc_id)
    if value:
        return int(value[1])

    return 0


def estimate_location(
    loc_id: int,
    target: datetime,
    ctx: Dict[str, object],
) -> Dict[str, float]:
    max_caps = ctx["max_caps"]
    max_cap = max_caps.get(loc_id, 0)
    if max_cap <= 0:
        return {
            "countP10": 0.0,
            "countP50": 0.0,
            "countP90": 0.0,
            "sampleCount": 0.0,
        }

    ratio = None
    model = ctx.get("model")
    loc_data = ctx["loc_data"]
    loc_samples = ctx["loc_samples"]
    loc_entry = loc_data.get(loc_id)
    onehot_vec = ctx["onehot"].get(loc_id)

    if (
        model is not None
        and loc_entry is not None
        and onehot_vec is not None
        and loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC
        and not loc_entry.get("is_stale")
    ):
        weather_source = ctx.get("weather_series")
        features = np.array(
            [build_features(target, loc_entry, onehot_vec, weather_source=weather_source)],
            dtype=np.float32,
        )
        ratio = float(model.predict(xgb.DMatrix(features))[0])

    if ratio is None or math.isnan(ratio):
        avg_dow_hour = ctx["avg_dow_hour"]
        avg_hour = ctx["avg_hour"]
        avg_overall = ctx["avg_overall"]

        dow = target.weekday()
        hour = target.hour
        value = avg_dow_hour.get((loc_id, dow, hour))
        if value:
            ratio = value[0]
        else:
            value = avg_hour.get((loc_id, hour))
            if value:
                ratio = value[0]
            else:
                value = avg_overall.get(loc_id)
                ratio = value[0] if value else 0.0

    p10_ratio, p50_ratio, p90_ratio = interval_bounds(
        point_ratio=float(ratio),
        hour=target.hour,
        residual_profile=ctx.get("interval_profile"),
    )

    sample_count = get_sample_count(
        loc_id,
        target,
        ctx["avg_dow_hour"],
        ctx["avg_hour"],
        ctx["avg_overall"],
    )

    return {
        "countP10": p10_ratio * max_cap,
        "countP50": p50_ratio * max_cap,
        "countP90": p90_ratio * max_cap,
        "sampleCount": float(sample_count),
    }


def sum_max_caps(max_caps: Dict[int, int], loc_ids: Iterable[int]) -> int:
    return sum(max_caps.get(loc_id, 0) for loc_id in loc_ids)


def round_count(value: float) -> int:
    return max(0, int(round(value)))


def to_hour_payload(
    target: datetime,
    sum_p10: float,
    sum_p50: float,
    sum_p90: float,
    category_max: int,
    samples: int,
) -> Dict[str, object]:
    expected_p10 = round_count(sum_p10)
    expected = round_count(sum_p50)
    expected_p90 = round_count(sum_p90)

    pct = round(min(expected / category_max, 1.0), 4) if category_max else None
    pct_p10 = round(min(expected_p10 / category_max, 1.0), 4) if category_max else None
    pct_p90 = round(min(expected_p90 / category_max, 1.0), 4) if category_max else None

    return {
        "hour": target.hour,
        "hourStart": target.isoformat(),
        "expectedCount": expected,
        "expectedPct": pct,
        "expectedCountP10": expected_p10,
        "expectedCountP90": expected_p90,
        "expectedPctP10": pct_p10,
        "expectedPctP90": pct_p90,
        "sampleCount": samples,
    }


def safe_parse_hour_start(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def latest_live_count_for_location(loc_entry: Optional[Dict[str, object]], now: datetime) -> Optional[float]:
    if not loc_entry:
        return None
    raw_times = loc_entry.get("raw_times") or []
    raw_values = loc_entry.get("raw_values") or []
    if not raw_times or not raw_values:
        return None

    ts = raw_times[-1]
    value = raw_values[-1]
    if ts is None or value is None:
        return None

    age_min = (now - ts).total_seconds() / 60.0
    if age_min < 0:
        age_min = 0.0
    if age_min > SPIKE_AWARE_MAX_AGE_MIN:
        return None

    return max(0.0, float(value))


def category_live_total(
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    now: datetime,
) -> Tuple[float, int]:
    total = 0.0
    observed_locs = 0

    for loc_id in loc_ids:
        live = latest_live_count_for_location(loc_data.get(loc_id), now)
        if live is None:
            continue
        total += live
        observed_locs += 1

    return total, observed_locs


def max_allowed_category_count(category_max: int) -> float:
    if not category_max:
        return float("inf")
    return max(float(category_max), float(category_max) * SPIKE_AWARE_MAX_CAP_MULTIPLIER)


def spike_weight(hours_ahead: int) -> float:
    if hours_ahead <= 0:
        return 1.0
    return math.exp(-SPIKE_AWARE_DECAY * max(0, hours_ahead - 1))


def apply_spike_adjustment_to_category_hours(
    day_hours: List[Dict[str, object]],
    category_max: int,
    live_total: float,
    now: datetime,
) -> bool:
    if not SPIKE_AWARE_ENABLED or not day_hours:
        return False

    parsed_starts = [safe_parse_hour_start(item.get("hourStart")) for item in day_hours]
    current_idx = None
    for idx, dt in enumerate(parsed_starts):
        if dt is None:
            continue
        if dt <= now:
            current_idx = idx
        else:
            break

    if current_idx is None:
        return False

    baseline_now = float(day_hours[current_idx].get("expectedCount", 0))
    drift = float(live_total) - baseline_now
    if abs(drift) < 1.0:
        return False

    max_allowed = max_allowed_category_count(category_max)
    adjusted = False

    for idx in range(current_idx + 1, len(day_hours)):
        dt = parsed_starts[idx]
        if dt is None:
            continue
        hours_ahead = idx - current_idx
        if hours_ahead > SPIKE_AWARE_HORIZON_HOURS:
            break

        weight = spike_weight(hours_ahead)
        delta = drift * weight
        if abs(delta) < 0.5:
            continue

        point = day_hours[idx]
        raw_mid = float(point.get("expectedCount", 0))
        raw_p10 = float(point.get("expectedCountP10", raw_mid))
        raw_p90 = float(point.get("expectedCountP90", raw_mid))

        new_mid = max(0.0, min(raw_mid + delta, max_allowed))
        new_p10 = max(0.0, min(raw_p10 + delta, max_allowed))
        new_p90 = max(0.0, min(raw_p90 + delta, max_allowed))
        if new_p10 > new_p90:
            new_p10, new_p90 = new_p90, new_p10
        new_mid = min(max(new_mid, new_p10), new_p90)

        expected = round_count(new_mid)
        expected_p10 = round_count(new_p10)
        expected_p90 = round_count(new_p90)

        point["expectedCountRaw"] = int(point.get("expectedCount", 0))
        point["expectedCount"] = expected
        point["expectedCountP10"] = expected_p10
        point["expectedCountP90"] = expected_p90
        point["spikeAdjusted"] = True

        if category_max:
            point["expectedPct"] = round(min(expected / category_max, 1.0), 4)
            point["expectedPctP10"] = round(min(expected_p10 / category_max, 1.0), 4)
            point["expectedPctP90"] = round(min(expected_p90 / category_max, 1.0), 4)
        else:
            point["expectedPct"] = None
            point["expectedPctP10"] = None
            point["expectedPctP90"] = None

        adjusted = True

    return adjusted


def build_total_series_from_categories(
    day_categories: List[Dict[str, object]],
    targets: List[datetime],
    now: datetime,
) -> List[Dict[str, object]]:
    rows = []

    for idx, target in enumerate(targets):
        sum_p10 = 0
        sum_p50 = 0
        sum_p90 = 0
        samples = 0

        for category in day_categories:
            hours = category.get("hours", [])
            if idx >= len(hours):
                continue
            point = hours[idx]
            count = int(point.get("expectedCount", 0))
            count_p10 = int(point.get("expectedCountP10", count))
            count_p90 = int(point.get("expectedCountP90", count))

            sum_p10 += count_p10
            sum_p50 += count
            sum_p90 += count_p90
            samples += int(point.get("sampleCount", 0))

        rows.append(
            {
                "hour": target.hour,
                "hourStart": target.isoformat(),
                "expectedTotal": round_count(sum_p50),
                "expectedTotalP10": round_count(sum_p10),
                "expectedTotalP50": round_count(sum_p50),
                "expectedTotalP90": round_count(sum_p90),
                "sampleCount": samples,
                "isFuture": target > now,
            }
        )

    return rows


def build_category_hours_for_targets(
    loc_ids: Iterable[int],
    targets: List[datetime],
    category_max: int,
    ctx: Dict[str, object],
) -> List[Dict[str, object]]:
    outputs = []

    for target in targets:
        sum_p10 = 0.0
        sum_p50 = 0.0
        sum_p90 = 0.0
        samples = 0

        for loc_id in loc_ids:
            result = estimate_location(
                loc_id,
                target,
                ctx,
            )
            sum_p10 += result["countP10"]
            sum_p50 += result["countP50"]
            sum_p90 += result["countP90"]
            samples += int(result["sampleCount"])

        outputs.append(
            to_hour_payload(
                target=target,
                sum_p10=sum_p10,
                sum_p50=sum_p50,
                sum_p90=sum_p90,
                category_max=category_max,
                samples=samples,
            )
        )

    return outputs


def get_targets_for_date(day_date: date) -> List[datetime]:
    start_hour = max(0, min(23, FORECAST_DAY_START_HOUR))
    end_hour = max(0, min(23, FORECAST_DAY_END_HOUR))
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour

    targets = []
    for hour in range(start_hour, end_hour + 1):
        naive = datetime(day_date.year, day_date.month, day_date.day, hour, 0, 0)
        targets.append(TZ.localize(naive))
    return targets


def parse_weather_hourly_payload(
    payload: Dict[str, object],
    min_dt: Optional[datetime] = None,
    max_dt: Optional[datetime] = None,
) -> Dict[str, object]:
    hourly = payload.get("hourly") or {}
    times_raw = hourly.get("time") or []
    if not times_raw:
        return {"times": [], "map": {}}

    weather_map: Dict[datetime, Dict[str, float]] = {}
    for idx, ts_raw in enumerate(times_raw):
        try:
            parsed = datetime.fromisoformat(str(ts_raw))
        except Exception:
            continue

        if parsed.tzinfo is None:
            parsed = TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(TZ)

        if min_dt is not None and parsed < min_dt:
            continue
        if max_dt is not None and parsed > max_dt:
            continue

        row: Dict[str, float] = {}
        for key, api_key in WEATHER_API_HOURLY_MAP.items():
            series = hourly.get(api_key) or []
            value = series[idx] if idx < len(series) else None
            numeric = to_float_or_none(value)
            if numeric is not None:
                row[key] = numeric

        if row:
            weather_map[parsed] = row

    times = sorted(weather_map.keys())
    return {"times": times, "map": weather_map}


def merge_weather_series(*series_list: Dict[str, object]) -> Dict[str, object]:
    merged_map: Dict[datetime, Dict[str, float]] = {}

    for series in series_list:
        weather_map = series.get("map", {})
        if not isinstance(weather_map, dict):
            continue
        for ts, row in weather_map.items():
            if not isinstance(ts, datetime) or not isinstance(row, dict):
                continue
            combined = merged_map.get(ts, {}).copy()
            combined.update(row)
            merged_map[ts] = combined

    times = sorted(merged_map.keys())
    return {"times": times, "map": merged_map}


def weather_history_start(loc_data: Dict[int, Dict[str, object]], now: datetime) -> datetime:
    floor_start = now - timedelta(days=max(1, WEATHER_HISTORY_MAX_DAYS))
    earliest = None
    for data in loc_data.values():
        bucket_times = data.get("bucket_times", [])
        if not bucket_times:
            continue
        first = bucket_times[0]
        if earliest is None or first < earliest:
            earliest = first

    if earliest is None:
        return floor_start

    return max(earliest - timedelta(hours=3), floor_start)


def fetch_weather_history_series(start_dt: datetime, end_dt: datetime) -> Dict[str, object]:
    if end_dt < start_dt:
        return {"times": [], "map": {}}

    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join(WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": TZ_NAME,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
    }

    try:
        resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        return parse_weather_hourly_payload(
            payload,
            min_dt=start_dt,
            max_dt=end_dt + timedelta(hours=3),
        )
    except Exception:
        return {"times": [], "map": {}}


def fetch_weather_forecast_series(now: datetime) -> Dict[str, object]:
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join(WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": TZ_NAME,
        "forecast_days": max(1, WEATHER_FORECAST_DAYS),
        "past_days": 2,
    }

    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return parse_weather_hourly_payload(payload, min_dt=now - timedelta(hours=12))
    except Exception:
        return {"times": [], "map": {}}


def build_weather_hours_for_targets(
    targets: List[datetime],
    weather_series: Dict[str, object],
) -> List[Dict[str, object]]:
    times = weather_series.get("times", [])
    weather_map = weather_series.get("map", {})
    rows = []

    for target in targets:
        def value(key: str):
            val = weather_value_at_or_before(times, weather_map, target, key)
            if math.isnan(val):
                return None
            if key == "weather_code":
                return int(round(val))
            return round(float(val), 2)

        rows.append(
            {
                "hour": target.hour,
                "hourStart": target.isoformat(),
                "tempC": value("temp_c"),
                "feelsLikeC": value("feels_like_c"),
                "precipMm": value("precip_mm"),
                "rainMm": value("rain_mm"),
                "snowCm": value("snow_cm"),
                "windMps": value("wind_mps"),
                "windGustMps": value("wind_gust_mps"),
                "humidityPct": value("humidity_pct"),
                "weatherCode": value("weather_code"),
            }
        )

    return rows


def build_weather_day_summary(weather_hours: List[Dict[str, object]]) -> Dict[str, object]:
    temps = [row["tempC"] for row in weather_hours if row.get("tempC") is not None]
    feels = [row["feelsLikeC"] for row in weather_hours if row.get("feelsLikeC") is not None]
    precip = [row["precipMm"] for row in weather_hours if row.get("precipMm") is not None]
    rain = [row["rainMm"] for row in weather_hours if row.get("rainMm") is not None]
    snow = [row["snowCm"] for row in weather_hours if row.get("snowCm") is not None]
    wind = [row["windMps"] for row in weather_hours if row.get("windMps") is not None]
    gust = [row["windGustMps"] for row in weather_hours if row.get("windGustMps") is not None]
    humidity = [row["humidityPct"] for row in weather_hours if row.get("humidityPct") is not None]
    weather_codes = [row["weatherCode"] for row in weather_hours if row.get("weatherCode") is not None]

    return {
        "avgTempC": round(sum(temps) / len(temps), 2) if temps else None,
        "avgFeelsLikeC": round(sum(feels) / len(feels), 2) if feels else None,
        "totalPrecipMm": round(sum(precip), 2) if precip else None,
        "totalRainMm": round(sum(rain), 2) if rain else None,
        "totalSnowCm": round(sum(snow), 2) if snow else None,
        "maxWindMps": round(max(wind), 2) if wind else None,
        "maxWindGustMps": round(max(gust), 2) if gust else None,
        "avgHumidityPct": round(sum(humidity) / len(humidity), 2) if humidity else None,
        "weatherCodes": sorted(set(int(code) for code in weather_codes)) if weather_codes else [],
    }


def normalize_series(series: List[Dict[str, object]]) -> List[Tuple[datetime, int, int]]:
    entries: List[Tuple[datetime, int, int]] = []
    for item in series:
        hour_start = item.get("hourStart")
        if not hour_start:
            continue
        entries.append(
            (
                datetime.fromisoformat(hour_start),
                int(item.get("expectedTotal", 0)),
                int(item.get("sampleCount", 0)),
            )
        )
    entries.sort(key=lambda row: row[0])
    return entries


def compute_range_metrics(
    series_entries: List[Tuple[datetime, int, int]],
    start: datetime,
    end: datetime,
) -> Tuple[int, float, int, int]:
    total_sum = 0
    hours = 0
    min_samples = None

    for dt, expected, samples in series_entries:
        if start <= dt < end:
            total_sum += expected
            hours += 1
            min_samples = samples if min_samples is None else min(min_samples, samples)

    if hours == 0:
        hours = max(1, int(round((end - start).total_seconds() / 3600)))

    return total_sum, round(total_sum / hours, 2), min_samples or 0, hours


def merge_windows_with_series(
    windows: List[Dict[str, object]],
    series: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not windows:
        return []

    ranges = []
    for window in windows:
        if not window or not window.get("start") or not window.get("end"):
            continue
        ranges.append(
            (
                datetime.fromisoformat(window["start"]),
                datetime.fromisoformat(window["end"]),
            )
        )

    if not ranges:
        return []

    ranges.sort(key=lambda row: row[0])
    merged_ranges = []
    cur_start, cur_end = ranges[0]
    allowed_gap = timedelta(hours=1)

    for start, end in ranges[1:]:
        if start <= (cur_end + allowed_gap):
            if end > cur_end:
                cur_end = end
        else:
            merged_ranges.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged_ranges.append((cur_start, cur_end))

    series_entries = normalize_series(series)
    merged = []
    for start, end in merged_ranges:
        total_sum, avg, min_samples, hours = compute_range_metrics(series_entries, start, end)
        merged.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "startHour": start.hour,
                "endHour": end.hour,
                "windowHours": hours,
                "expectedTotal": total_sum,
                "expectedAvg": avg,
                "sampleCountMin": min_samples,
            }
        )

    return merged


def smooth_values(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, window)
    if window == 1:
        return values[:]

    half = window // 2
    smoothed = []
    n = len(values)

    for idx in range(n):
        start = max(0, idx - half)
        end = min(n, idx + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def detect_extrema_indices(
    series: List[Dict[str, object]],
    smoothed: List[float],
    mean: float,
    std: float,
    threshold: float,
    mode: str,
) -> List[int]:
    if not series or not smoothed:
        return []

    indices = []
    n = len(smoothed)

    for idx in range(n):
        if series[idx]["sampleCount"] < MIN_SAMPLES_PER_HOUR_TOTAL:
            continue

        left = smoothed[idx - 1] if idx - 1 >= 0 else smoothed[idx]
        right = smoothed[idx + 1] if idx + 1 < n else smoothed[idx]

        if mode == "peak":
            if smoothed[idx] < left or smoothed[idx] < right:
                continue
            if smoothed[idx] < mean + threshold * std:
                continue
            indices.append(idx)

        if mode == "low":
            if smoothed[idx] > left or smoothed[idx] > right:
                continue
            if smoothed[idx] > mean - threshold * std:
                continue
            indices.append(idx)

    return indices


def window_indices_for_center(index: int, window: int, length: int) -> Tuple[int, int]:
    window = max(1, min(window, length))
    half = window // 2
    start = max(0, index - half)
    if start + window > length:
        start = length - window
    return start, start + window


def build_windows_for_indices(
    series: List[Dict[str, object]],
    indices: List[int],
    window_hours: int,
) -> List[Dict[str, object]]:
    if not series or not indices:
        return []

    windows = {}
    n = len(series)

    for idx in indices:
        start_idx, end_idx = window_indices_for_center(idx, window_hours, n)
        window_slice = series[start_idx:end_idx]
        if not window_slice:
            continue

        total_sum = sum(item["expectedTotal"] for item in window_slice)
        min_samples = min(item["sampleCount"] for item in window_slice)

        start_iso = window_slice[0]["hourStart"]
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = start_dt + timedelta(hours=len(window_slice))

        windows[start_iso] = {
            "start": start_iso,
            "end": end_dt.isoformat(),
            "startHour": window_slice[0]["hour"],
            "endHour": end_dt.hour,
            "windowHours": len(window_slice),
            "expectedTotal": total_sum,
            "expectedAvg": round(total_sum / len(window_slice), 2),
            "sampleCountMin": min_samples,
            "centerHour": series[idx]["hour"],
        }

    return sorted(windows.values(), key=lambda item: item["start"])


def find_peak_and_low_windows(
    series: List[Dict[str, object]],
    window_hours: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not series:
        return [], []

    values = [item["expectedTotal"] for item in series]
    smoothed = smooth_values(values, SMOOTH_WINDOW)
    mean, std = mean_std(smoothed)

    peak_indices = detect_extrema_indices(series, smoothed, mean, std, PEAK_Z_THRESHOLD, "peak")
    low_indices = detect_extrema_indices(series, smoothed, mean, std, LOW_Z_THRESHOLD, "low")

    if not peak_indices:
        peak_indices = detect_extrema_indices(series, smoothed, mean, std, 0.0, "peak")
    if not low_indices:
        low_indices = detect_extrema_indices(series, smoothed, mean, std, 0.0, "low")

    return (
        build_windows_for_indices(series, peak_indices, window_hours),
        build_windows_for_indices(series, low_indices, window_hours),
    )


def build_forecast():
    now = datetime.now(TZ)
    week_dates = [(now + timedelta(days=offset)).date() for offset in range(7)]

    conn = db_connect()
    try:
        (
            loc_data,
            avg_dow_hour,
            avg_hour,
            avg_overall,
            max_caps,
            loc_samples,
            quality,
        ) = load_history(conn)
    finally:
        conn.close()

    expected_loc_ids = all_location_ids()
    onehot = build_onehot(expected_loc_ids)

    history_start = weather_history_start(loc_data, now)
    weather_history_series = fetch_weather_history_series(history_start, now)
    future_weather_series = fetch_weather_forecast_series(now)
    weather_series = merge_weather_series(weather_history_series, future_weather_series)
    quality["weatherHistoryHours"] = len(weather_history_series.get("times", []))
    quality["weatherForecastHours"] = len(future_weather_series.get("times", []))
    quality["weatherMergedHours"] = len(weather_series.get("times", []))
    quality["weatherAvailable"] = bool(weather_series.get("times"))

    model, model_meta, model_status, run_metrics = prepare_model(
        now=now,
        expected_loc_ids=expected_loc_ids,
        onehot=onehot,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
    )

    interval_profile = model_meta.get("residualProfile") if model_meta else None

    ctx = {
        "now": now,
        "model": model,
        "interval_profile": interval_profile,
        "weather_series": weather_series,
        "loc_data": loc_data,
        "onehot": onehot,
        "avg_dow_hour": avg_dow_hour,
        "avg_hour": avg_hour,
        "avg_overall": avg_overall,
        "loc_samples": loc_samples,
        "max_caps": max_caps,
    }

    facilities_payload = []
    spike_adjusted_categories = 0

    for facility_id, facility in FACILITIES.items():
        weekly_forecast = []

        for day_date in week_dates:
            targets = get_targets_for_date(day_date)
            day_weather = build_weather_hours_for_targets(targets, weather_series)
            day_weather_summary = build_weather_day_summary(day_weather)
            day_categories = []

            for category in facility["categories"]:
                category_max = sum_max_caps(max_caps, category["location_ids"])
                day_hours = build_category_hours_for_targets(
                    loc_ids=category["location_ids"],
                    targets=targets,
                    category_max=category_max,
                    ctx=ctx,
                )

                live_total, observed_locs = category_live_total(
                    category["location_ids"],
                    loc_data,
                    now,
                )
                if observed_locs > 0:
                    if apply_spike_adjustment_to_category_hours(
                        day_hours=day_hours,
                        category_max=category_max,
                        live_total=live_total,
                        now=now,
                    ):
                        spike_adjusted_categories += 1

                day_categories.append(
                    {
                        "key": category["key"],
                        "title": category["title"],
                        "maxCapacity": category_max or None,
                        "hours": day_hours,
                    }
                )

            totals_day = build_total_series_from_categories(
                day_categories=day_categories,
                targets=targets,
                now=now,
            )

            avoid_windows_day, best_windows_day = find_peak_and_low_windows(
                totals_day,
                PEAK_WINDOW_HOURS,
            )
            avoid_windows_day = merge_windows_with_series(avoid_windows_day, totals_day)
            best_windows_day = merge_windows_with_series(best_windows_day, totals_day)

            weekly_forecast.append(
                {
                    "dayName": day_date.strftime("%A"),
                    "date": day_date.isoformat(),
                    "weatherHours": day_weather,
                    "weatherSummary": day_weather_summary,
                    "categories": day_categories,
                    "avoidWindows": avoid_windows_day,
                    "bestWindows": best_windows_day,
                }
            )

        facilities_payload.append(
            {
                "facilityId": facility_id,
                "facilityName": facility["name"],
                "weeklyForecast": weekly_forecast,
            }
        )

    quality["locationsModeled"] = sum(
        1
        for loc_id, data in loc_data.items()
        if loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )
    quality["spikeAwareEnabled"] = SPIKE_AWARE_ENABLED
    quality["spikeAwareCategoriesAdjusted"] = spike_adjusted_categories

    val_mae_for_payload = model_meta.get("valMae") if model_meta else run_metrics.get("val_mae")
    precision_pct = None
    if val_mae_for_payload is not None:
        precision_pct = max(0.0, min(100.0, (1.0 - float(val_mae_for_payload)) * 100.0))

    payload = {
        "generatedAt": now.isoformat(),
        "timezone": TZ_NAME,
        "model": "xgboost",
        "peakWindowHours": max(1, PEAK_WINDOW_HOURS),
        "peakDetection": {
            "smoothWindow": max(1, SMOOTH_WINDOW),
            "peakZ": PEAK_Z_THRESHOLD,
            "lowZ": LOW_Z_THRESHOLD,
        },
        "modelInfo": {
            "status": model_status,
            "trainedAt": model_meta.get("trainedAt") if model_meta else None,
            "trainRows": model_meta.get("trainRows") if model_meta else run_metrics.get("train_rows"),
            "valRows": model_meta.get("valRows") if model_meta else run_metrics.get("val_rows"),
            "valMae": val_mae_for_payload,
            "valRmse": model_meta.get("valRmse") if model_meta else run_metrics.get("val_rmse"),
            "precisionPct": precision_pct,
            "retrainHours": MODEL_RETRAIN_HOURS,
            "guardrailMaxMaeDegrade": MODEL_GUARDRAIL_MAX_MAE_DEGRADE,
            "spikeAwareEnabled": SPIKE_AWARE_ENABLED,
            "spikeAwareHorizonHours": SPIKE_AWARE_HORIZON_HOURS,
            "spikeAwareMaxAgeMin": SPIKE_AWARE_MAX_AGE_MIN,
        },
        "dataQuality": quality,
        "facilities": facilities_payload,
    }

    return payload


def write_forecast(payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(FORECAST_JSON_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = FORECAST_JSON_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, FORECAST_JSON_PATH)


def main() -> int:
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        payload = build_forecast()
        write_forecast(payload)

        facilities_count = len(payload.get("facilities", []))
        model_info = payload.get("modelInfo", {})
        status = model_info.get("status")
        val_mae = model_info.get("valMae")
        val_rmse = model_info.get("valRmse")
        val_rows = model_info.get("valRows")
        precision_pct = model_info.get("precisionPct")

        metric_part = ""
        if val_mae is not None and val_rmse is not None:
            metric_part = f" | valRows {val_rows} | valMAE {float(val_mae):.4f} | valRMSE {float(val_rmse):.4f}"
        if precision_pct is not None:
            metric_part += f" | precision {float(precision_pct):.2f}%"
        else:
            metric_part += " | precision n/a"

        print(
            f"{start} OK: facilities {facilities_count} | modelStatus {status}{metric_part}"
        )
        return 0
    except Exception as exc:
        print(start, "ERROR:", "{}: {}".format(type(exc).__name__, exc))
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
