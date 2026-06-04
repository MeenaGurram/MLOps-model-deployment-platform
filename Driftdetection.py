import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference = pd.read_csv(
    "data/reference.csv"
)

current = pd.read_csv(
    "data/current.csv"
)

report = Report(
    metrics=[DataDriftPreset()]
)

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html(
    "reports/drift_report.html"
)