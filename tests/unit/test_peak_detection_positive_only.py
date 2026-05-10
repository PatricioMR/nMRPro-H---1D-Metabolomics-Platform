import numpy as np

from processing.peak_detection import detect_peaks


def test_negative_valley_is_not_detected_as_peak():
    ppm = np.linspace(0.0, 1.0, 800)
    signal = -1.0 * np.exp(-((ppm - 0.50) ** 2) / 0.0005)

    result = detect_peaks(
        ppm,
        signal,
        noise_std=0.02,
        min_snr=3.0,
        baseline_correct=False,
    )

    assert result == []


def test_positive_peak_survives_when_negative_valley_is_present():
    ppm = np.linspace(0.0, 1.0, 800)
    positive = 1.2 * np.exp(-((ppm - 0.32) ** 2) / 0.0004)
    negative = -0.8 * np.exp(-((ppm - 0.68) ** 2) / 0.0005)
    signal = positive + negative

    result = detect_peaks(
        ppm,
        signal,
        noise_std=0.02,
        min_snr=3.0,
        baseline_correct=False,
    )

    assert len(result) == 1
    assert abs(result[0]['ppm'] - 0.32) < 0.01
    assert result[0]['height'] > 0