"""
coral_capture.py — shared serial capture + analysis utility for Coral E1/E2.
Handles USB disconnect/reconnect caused by nRST button press.
"""

import serial
import serial.tools.list_ports
import time
import numpy as np

BAUD_RATE    = 115200
READ_TIMEOUT = 120


def find_coral_port():
    for p in serial.tools.list_ports.comports():
        if "ttyACM" in p.device or "ttyUSB" in p.device:
            return p.device
    return None


def wait_for_port(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        port = find_coral_port()
        if port:
            return port
        time.sleep(0.2)
    raise RuntimeError("Coral port did not reappear within 30s after reset")


def capture_inferences(port, n_expected=2000, timeout=READ_TIMEOUT):
    """
    Prompt user to press nRST, wait for USB reconnect, capture infer lines.
    Returns list of (iteration, class_id, score) tuples, or empty list on failure.
    """
    results = []

    print("    [Press nRST on Coral now — wait 1 full second before releasing...]", flush=True)

    # Wait up to 15s for disconnect
    deadline = time.time() + 15
    disconnected = False
    while time.time() < deadline:
        if find_coral_port() is None:
            print("    [USB disconnect detected — waiting for reconnect...]", flush=True)
            disconnected = True
            break
        time.sleep(0.1)

    if not disconnected:
        print("    [No disconnect detected — press nRST more firmly for ~1 second]", flush=True)
        # Wait a bit longer in case reset was fast
        time.sleep(1.0)

    # Wait for port to reappear
    try:
        port = wait_for_port(timeout=30)
    except RuntimeError as e:
        print(f"    [ERROR: {e}]", flush=True)
        return results

    print(f"    [Coral reconnected on {port} — waiting for firmware boot...]", flush=True)
    time.sleep(3.0)   # give firmware time to boot and start printing

    deadline = time.time() + timeout
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1.0) as ser:
            ser.reset_input_buffer()

            while time.time() < deadline:
                try:
                    raw = ser.readline()
                except serial.SerialException:
                    break

                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()

                if line.startswith("infer,"):
                    parts = line.split(",")
                    if len(parts) == 4:
                        try:
                            iteration = int(parts[1])
                            class_id  = int(parts[2])
                            score     = float(parts[3])
                            results.append((iteration, class_id, score))
                            if len(results) % 500 == 0:
                                print(f"    [{len(results)}/{n_expected} captured]", flush=True)
                        except ValueError:
                            pass

                if "E0 Infer Baseline End" in line:
                    print("    [firmware complete]", flush=True)
                    break

                if len(results) >= n_expected:
                    break

    except serial.SerialException as e:
        print(f"    [Serial error: {e}]", flush=True)

    return results


def compute_metrics(rows, ref_class=None, ref_score=None):
    if not rows:
        raise RuntimeError("No inference rows captured")

    if ref_class is None:
        ref_class = rows[0][1]
    if ref_score is None:
        ref_score = rows[0][2]

    violations = 0
    deltas = []

    for _, class_id, score in rows:
        if class_id != ref_class:
            violations += 1
        deltas.append(abs(score - ref_score))

    n = len(rows)
    return {
        "n":          n,
        "ster":       violations / n,
        "acc":        1.0 - (violations / n),
        "d_mean":     float(np.mean(deltas)),
        "d_p99":      float(np.percentile(deltas, 99)),
        "violations": violations,
        "ref_class":  ref_class,
        "ref_score":  ref_score,
    }
