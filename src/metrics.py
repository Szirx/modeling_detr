from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MetricCollection


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'mAP': MeanAveragePrecision(**kwargs),
        },
    )
