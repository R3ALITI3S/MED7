from collections import deque
import numpy as np
import time


class BlinkDetector:
    """
    Real-time blink detector. Based on blinkdetection.py but for realtime now

    Uses pupil diameter from both eyes.
    Detects:
      - Simultaneous pupil diameter reductions
      - Missing pupil measurements from both eyes

    Designed for the approx 50 Hz input.
    """

    def __init__(
        self,
        threshold=0.80,
        baseline_window=150,
        min_duration=0.03,
        max_duration=0.30,
        merge_gap=0.04
    ):

        self.threshold = threshold
        self.baseline_window = baseline_window

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_gap = merge_gap

        self.left_history = deque(maxlen=baseline_window)
        self.right_history = deque(maxlen=baseline_window)

        self.blinking = False
        self.blink_start = None
        self.last_candidate = None

        self.blink_count = 0
        self.last_blink_duration = 0

    def update(self, left, right, timestamp=None):

        if timestamp is None:
            timestamp = time.monotonic()

        # Convert invalid values into NaN
        left = self._valid_pupil(left)
        right = self._valid_pupil(right)

        # Calculate baselines BEFORE adding current measurements
        baseline_left = self._baseline(self.left_history)
        baseline_right = self._baseline(self.right_history)

        # Determine whether both pupils indicate closure
        both_missing = np.isnan(left) and np.isnan(right)

        left_closed = (
            not np.isnan(left)
            and baseline_left is not None
            and left / baseline_left < self.threshold
        )

        right_closed = (
            not np.isnan(right)
            and baseline_right is not None
            and right / baseline_right < self.threshold
        )

        candidate = both_missing or (left_closed and right_closed)

        # Update histories only with valid measurements
        if not np.isnan(left):
            self.left_history.append(left)

        if not np.isnan(right):
            self.right_history.append(right)

        # Start a candidate blink
        if candidate:

            if not self.blinking:
                self.blinking = True
                self.blink_start = timestamp

            self.last_candidate = timestamp

        # Candidate has ended
        elif self.blinking:

            if self.last_candidate is not None:
                gap = timestamp - self.last_candidate

                # Allow short gaps between candidate samples
                if gap > self.merge_gap:

                    duration = self.last_candidate - self.blink_start

                    self.blinking = False

                    if self.min_duration <= duration <= self.max_duration:

                        self.blink_count += 1
                        self.last_blink_duration = duration

                        return {
                            "blink_detected": True,
                            "blink_count": self.blink_count,
                            "duration": duration
                        }

                    self.blink_start = None
                    self.last_candidate = None

        return {
            "blink_detected": False,
            "blink_count": self.blink_count,
            "duration": self.last_blink_duration
        }

    @staticmethod
    def _valid_pupil(value):

        try:
            value = float(value)

            if not np.isfinite(value) or value <= 0:
                return np.nan

            return value

        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _baseline(history):

        if len(history) < 30:
            return None

        return float(np.median(history))
