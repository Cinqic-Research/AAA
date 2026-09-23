"""Four datasets from the Monash Time Series Forecasting Archive, fetched and verified.

Source: Godahewa, Bergmeir, Webb, Hyndman & Montero-Manso (2021), *Monash Time
Series Forecasting Archive*, NeurIPS Datasets and Benchmarks; data on Zenodo
under CC BY 4.0. Files are not redistributed in this repository: they are
downloaded on demand into a configurable data root and refused unless both the
Zenodo MD5 and the recorded SHA-256 of the archive and of the extracted
``.tsf`` match.

Selection rule (declared before any AAA model saw any Monash data), applied to
the dataset table published at forecastingdata.org on 2026-09-23:

1. a variant without missing values exists;
2. the shortest series has at least 700 observations (room for online learning);
3. the dataset has at most 2.5 million observations in total (local compute);
4. one dataset per sampling frequency among those left, choosing the one with
   the fewest series (ties alphabetical).

Result: Saugeen River Flow (daily), M4 Hourly (hourly), Australian Electricity
Demand (half-hourly), FRED-MD (monthly). No weekly, 10-minute or 4-second
dataset survives rules 2-3. Every series of each selected dataset is used.

Evaluation follows the archive's fixed-horizon protocol: the last ``h``
observations of each series are the test set, ``h`` from the ``.tsf`` header
(M4 Hourly: 48) or the paper's rule for non-competition datasets (daily 30,
monthly 12, half-hourly one week = 336); MASE uses the in-sample seasonal
naive scale at the *daily* seasonality for sub-daily data (hourly 24,
half-hourly 48), 7 for daily and 12 for monthly data, exactly as
``utils/error_calculator.R`` does with ``min(seasonality)``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CITATION = (
    "Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J. & Montero-Manso, P. (2021). Monash Time Series "
    "Forecasting Archive. NeurIPS Track on Datasets and Benchmarks. https://forecastingdata.org"
)
LICENSE = "CC-BY-4.0"


@dataclass(frozen=True)
class MonashDataset:
    name: str
    zenodo_record: int
    archive: str
    tsf: str
    md5: str
    archive_sha256: str
    tsf_sha256: str
    frequency: str
    horizon: int
    seasonality: tuple[float, ...]
    horizon_source: str


DATASETS: dict[str, MonashDataset] = {
    "saugeen": MonashDataset(
        "saugeen",
        4656058,
        "saugeenday_dataset.zip",
        "saugeenday_dataset.tsf",
        "25365107befc47a70fae017043993fd5",
        "061cb56ee6829d5686e5e08f1a61f4c0af682e32c37824e79926fc11148747df",
        "bfd2ad2d80d8f3b6cd55086716dc9047789ebd76e0c78c06bba7af5b812d95fc",
        "daily",
        30,
        (7.0,),
        "paper rule: 30 days for daily datasets",
    ),
    "m4_hourly": MonashDataset(
        "m4_hourly",
        4656589,
        "m4_hourly_dataset.zip",
        "m4_hourly_dataset.tsf",
        "3983df90f60db0f4e1f4dc3fca8ef565",
        "18085bd3c34e41cdc07441aa61c5610dac9e916b9489a6a381f8e89fd01c8a66",
        "f3c0112e09ce616777ead9e57c9d85499ebab329d41a851b4edcec2944c92eb8",
        "hourly",
        48,
        (24.0, 168.0, 8766.0),
        ".tsf @horizon 48 (competition horizon)",
    ),
    "aus_elec_demand": MonashDataset(
        "aus_elec_demand",
        4659727,
        "australian_electricity_demand_dataset.zip",
        "australian_electricity_demand_dataset.tsf",
        "c9ec3ee8e81cd78d9edd2a29a7a2640b",
        "b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a",
        "73af0efd30a102d63b7e4ae2972ced64d4172eba20cfd4614016a8c83a748c7f",
        "half_hourly",
        336,
        (48.0, 336.0, 17532.0),
        "paper rule: one week for half-hourly datasets",
    ),
    "fred_md": MonashDataset(
        "fred_md",
        4654833,
        "fred_md_dataset.zip",
        "fred_md_dataset.tsf",
        "339cc7c28d9e6358bd6ed4115f1b166d",
        "305c0edd2b5e97159c6339be4990ea97fdf86772c1edef2a0dfc836bf29f45c3",
        "b62383200451cd400e372583f76dfd4054d7045fe1a7ee994d97734c8cf62f3d",
        "monthly",
        12,
        (12.0,),
        "paper rule: 12 months for monthly datasets",
    ),
}

PUBLISHED_MASE: dict[str, dict[str, float]] = {
    # forecastingdata.org results table, mean MASE, read 2026-09-23 (page sha256 ef7a4916...f8d8).
    "saugeen": {
        "SES": 1.426,
        "Theta": 1.425,
        "TBATS": 1.477,
        "ETS": 2.036,
        "(DHR-)ARIMA": 1.485,
        "PR": 1.674,
        "CatBoost": 1.411,
        "FFNN": 1.524,
        "DeepAR": 1.56,
        "N-BEATS": 1.852,
        "WaveNet": 1.471,
        "Transformer": 1.861,
    },
    "m4_hourly": {
        "SES": 11.607,
        "Theta": 11.524,
        "TBATS": 2.663,
        "ETS": 26.69,
        "(DHR-)ARIMA": 13.557,
        "PR": 1.662,
        "CatBoost": 1.771,
        "FFNN": 2.862,
        "DeepAR": 2.145,
        "N-BEATS": 2.247,
        "WaveNet": 1.68,
        "Transformer": 8.84,
    },
    "aus_elec_demand": {
        "SES": 1.857,
        "Theta": 1.867,
        "TBATS": 1.174,
        "ETS": 5.663,
        "(DHR-)ARIMA": 2.574,
        "PR": 0.78,
        "CatBoost": 0.705,
        "FFNN": 1.222,
        "DeepAR": 1.591,
        "N-BEATS": 1.014,
        "WaveNet": 1.102,
        "Transformer": 1.113,
    },
    "fred_md": {
        "SES": 0.617,
        "Theta": 0.698,
        "TBATS": 0.502,
        "ETS": 0.468,
        "(DHR-)ARIMA": 0.533,
        "PR": 8.827,
        "CatBoost": 0.947,
        "FFNN": 0.601,
        "DeepAR": 0.64,
        "N-BEATS": 0.604,
        "WaveNet": 0.806,
        "Transformer": 1.823,
    },
}
PUBLISHED_NOTE = (
    "published baselines are fitted offline per series (local) or across series (global) on the same training "
    "split and forecast the same test horizon with the same MASE; AAA models learn online over the training "
    "split and forecast recursively. The results-table row labelled 'Aus. Elecdemand' is mapped to the "
    "Australian Electricity Demand dataset by name only; that mapping is not verified."
)


class DataUnavailable(RuntimeError):
    """The dataset is not present in the data root and fetching was not permitted or failed."""


def data_root(explicit: str | Path | None = None) -> Path:
    """``explicit``, else ``$AAA_DATA_ROOT``, else ``~/.cache/aaa/external``. Never a hard-coded machine path."""

    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("AAA_DATA_ROOT")
    return Path(env) if env else Path.home() / ".cache" / "aaa" / "external"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(dataset: MonashDataset, root: Path, *, fetch: bool = False) -> Path:
    """Return the verified ``.tsf`` path, fetching from Zenodo only when ``fetch`` is true."""

    directory = root / "monash"
    tsf = directory / dataset.tsf
    archive = directory / dataset.archive
    if not tsf.exists():
        if not archive.exists():
            if not fetch:
                raise DataUnavailable(f"{dataset.name}: {archive} missing (run with fetch enabled)")
            directory.mkdir(parents=True, exist_ok=True)
            url = f"https://zenodo.org/records/{dataset.zenodo_record}/files/{dataset.archive}?download=1"
            temporary = archive.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(archive)
        if (
            hashlib.md5(archive.read_bytes()).hexdigest() != dataset.md5
            or _sha256(archive) != dataset.archive_sha256
        ):
            raise DataUnavailable(f"{dataset.name}: archive checksum mismatch; refusing to use it")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract(dataset.tsf, directory)
    if _sha256(tsf) != dataset.tsf_sha256:
        raise DataUnavailable(f"{dataset.name}: .tsf checksum mismatch; refusing to use it")
    return tsf


def parse_tsf(path: Path) -> dict[str, Any]:
    """Minimal ``.tsf`` reader: header attributes and one float array per series."""

    header: dict[str, str] = {}
    attributes: list[str] = []
    series: list[dict[str, Any]] = []
    in_data = False
    with path.open(encoding="cp1252", newline=None) as handle:  # the archive's own loader uses cp1252
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.startswith("@attribute"):
                    attributes.append(line.split()[1])
                elif line.startswith("@data"):
                    in_data = True
                elif line.startswith("@"):
                    key, _, value = line[1:].partition(" ")
                    header[key] = value.strip()
                continue
            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise ValueError(f"{path.name}: malformed series line")
            values = fields[-1].split(",")
            if any(value == "?" for value in values):
                raise ValueError(f"{path.name}: missing values in a dataset declared complete")
            series.append(
                {
                    "attributes": dict(zip(attributes, fields[:-1], strict=True)),
                    "values": np.asarray(values, dtype=float),
                }
            )
    return {"header": header, "attributes": attributes, "series": series}


@dataclass(frozen=True)
class SplitSeries:
    name: str
    training: np.ndarray
    test: np.ndarray


def load(
    name: str, *, root: str | Path | None = None, fetch: bool = False, role: str = "confirmation"
) -> dict[str, Any]:
    """Series split for ``role``.

    ``confirmation``: training = all but the last ``h``; test = last ``h`` (the
    archive's split). ``development``: the archive's *training* part is split
    again, holding out its own last ``h`` -- the archive's test observations
    are never read in development.
    """

    dataset = DATASETS[name]
    parsed = parse_tsf(ensure(dataset, data_root(root), fetch=fetch))
    header_horizon = parsed["header"].get("horizon")
    if header_horizon is not None and int(header_horizon) != dataset.horizon:
        raise ValueError(
            f"{name}: header horizon {header_horizon} disagrees with the declared {dataset.horizon}"
        )
    if parsed["header"].get("frequency") != dataset.frequency:
        raise ValueError(f"{name}: frequency {parsed['header'].get('frequency')} is not {dataset.frequency}")
    h = dataset.horizon
    splits = []
    for index, entry in enumerate(parsed["series"]):
        values = entry["values"]
        label = entry["attributes"].get("series_name", f"T{index + 1}")
        training, test = values[:-h], values[-h:]
        if role == "development":
            training, test = training[:-h], training[-h:]
        elif role != "confirmation":
            raise ValueError("role must be development or confirmation")
        splits.append(SplitSeries(label, training, test))
    return {
        "dataset": dataset,
        "series": splits,
        "role": role,
        "header": parsed["header"],
    }
