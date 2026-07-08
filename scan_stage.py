from pipython import GCSDevice, pitools, GCSError
import csv
import os
import math
 
 
SERIAL = '118061293'
AXIS = '1'
 
STEP_MM = 0.5
DWELL_SEC = 0.5
 
# Safety margins away from hard limits
LIMIT_MARGIN_MM = 0.5
 
# Gentle motion settings
APPROACH_STEP_MM = 0.5       # small steps for long moves
RETURN_STEP_MM = 0.5         # small steps on return
INTERSTEP_DWELL_SEC = 0.20   # pause between approach steps
WAIT_TIMEOUT_SEC = 120
 
log_csv = os.path.join(SCAN_DIR, "scan_log.csv")
 
 
def get_scalar_from_query_result(val, axis):
    if isinstance(val, dict):
        return float(val[axis])
    return float(val)
 
 
def wait_target_safe(pidevice, axis, timeout=WAIT_TIMEOUT_SEC, polldelay=0.05):
    pitools.waitontarget(
        pidevice,
        axes=axis,
        timeout=timeout,
        polldelay=polldelay
    )
 
 
def move_gently(pidevice, axis, target, step_mm=0.5, dwell_sec=0.2, timeout=WAIT_TIMEOUT_SEC):
    current = get_scalar_from_query_result(pidevice.qPOS(axis), axis)
    delta = target - current
 
    if abs(delta) < 1e-9:
        return current
 
    direction = 1.0 if delta > 0 else -1.0
    n_full_steps = int(abs(delta) // step_mm)
 
    for _ in range(n_full_steps):
        current += direction * step_mm
        pidevice.MOV(axis, current)
        wait_target_safe(pidevice, axis, timeout=timeout)
        sleep(dwell_sec)
 
    if abs(target - current) > 1e-9:
        pidevice.MOV(axis, target)
        wait_target_safe(pidevice, axis, timeout=timeout)
        sleep(dwell_sec)
 
    return get_scalar_from_query_result(pidevice.qPOS(axis), axis)
 
 
def frange_inclusive(start, stop, step):
    vals = []
    x = start
    if step <= 0:
        raise ValueError("step must be > 0")
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals
 
 
with open(log_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "position_mm",
        "frame_base",
        "frame_raw",
        "frame_bgsub",
        "bg_level",
        "width_px",
        "height_px",
    ])
 
    with GCSDevice() as pidevice:
        pidevice.ConnectUSB(serialnum=SERIAL)
        print("IDN:", pidevice.qIDN().strip())
 
        pitools.startup(
            pidevice,
            stages=None,
            refmodes='FRF',
            servostates=True
        )
 
        sleep(0.5)
 
        initial_pos = get_scalar_from_query_result(pidevice.qPOS(AXIS), AXIS)
        print(f"Initial position: {initial_pos:.6f} mm")
 
        # Query axis travel limits from controller
        try:
            tmin = get_scalar_from_query_result(pidevice.qTMN(AXIS), AXIS)
            tmax = get_scalar_from_query_result(pidevice.qTMX(AXIS), AXIS)
        except Exception as err:
            raise RuntimeError(
                "Could not query travel limits with qTMN/qTMX. "
                "Please verify this controller/stage supports those queries."
            ) from err
 
        scan_start = tmin + LIMIT_MARGIN_MM
        scan_end = tmax - LIMIT_MARGIN_MM
 
        if scan_end <= scan_start:
            raise RuntimeError(
                f"Invalid usable scan range after margin: start={scan_start}, end={scan_end}"
            )
 
        print(f"Reported travel range: {tmin:.6f} mm to {tmax:.6f} mm")
        print(f"Usable scan range:    {scan_start:.6f} mm to {scan_end:.6f} mm")
        print(f"Step size:            {STEP_MM:.6f} mm")
 
        positions = frange_inclusive(scan_start, scan_end, STEP_MM)
        if not positions:
            raise RuntimeError("No scan positions generated.")
 
        print(f"Number of scan points: {len(positions)}")
 
        try:
            print(f"\nGently moving to scan start {scan_start:.6f} mm")
            move_gently(
                pidevice,
                AXIS,
                scan_start,
                step_mm=APPROACH_STEP_MM,
                dwell_sec=INTERSTEP_DWELL_SEC,
                timeout=WAIT_TIMEOUT_SEC
            )
            print("At start:", pidevice.qPOS(AXIS))
 
            for pos in positions:
                current_pos = get_scalar_from_query_result(pidevice.qPOS(AXIS), AXIS)
 
                if abs(current_pos - pos) > 1e-6:
                    print(f"\nMoving gently to {pos:.3f} mm")
                    move_gently(
                        pidevice,
                        AXIS,
                        pos,
                        step_mm=min(APPROACH_STEP_MM, STEP_MM),
                        dwell_sec=INTERSTEP_DWELL_SEC,
                        timeout=WAIT_TIMEOUT_SEC
                    )
 
                reached = get_scalar_from_query_result(pidevice.qPOS(AXIS), AXIS)
                print(f"At position {reached:.3f} mm, waiting {DWELL_SEC:.2f} s...")
                sleep(DWELL_SEC)
 
                res = acquire_beam_frame(
                    gd_ctrl,
                    prefix=f"z_{reached:+08.3f}",
                    save_dir=SCAN_DIR
                )
                print("Captured frame:", os.path.basename(res["frame_path"]))
 
                writer.writerow([
                    f"{reached:.6f}",
                    os.path.basename(res["base"]),
                    os.path.basename(res["frame_path"]),
                    os.path.basename(res["bgsub_path"]),
                    f"{res['bg']:.6f}",
                    res["w"],
                    res["h"],
                ])
                f.flush()
 
        except GCSError as err:
            print("\nMotion error during scan:", err)
            print("Stopping scan and attempting gentle return to initial position.")
        finally:
            print(f"\nReturning gently to initial position {initial_pos:.6f} mm")
            try:
                move_gently(
                    pidevice,
                    AXIS,
                    initial_pos,
                    step_mm=RETURN_STEP_MM,
                    dwell_sec=INTERSTEP_DWELL_SEC,
                    timeout=WAIT_TIMEOUT_SEC
                )
                final_pos = get_scalar_from_query_result(pidevice.qPOS(AXIS), AXIS)
                print(f"Returned to: {final_pos:.6f} mm")
            except GCSError as err:
                print("Error while returning to initial position:", err)
 
print("Scan finished")
print("All data saved in:", SCAN_DIR)
print("Log file:", log_csv) import os