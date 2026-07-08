import datetime
 
 
SCAN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
 
# Parent folder next to your notebooks
SCAN_ROOT = os.path.join(os.getcwd(), "all_scans")
os.makedirs(SCAN_ROOT, exist_ok=True)
 
# One subfolder per run inside all_scans
SCAN_DIR = os.path.join(SCAN_ROOT, f"scan_{SCAN_TIMESTAMP}")
os.makedirs(SCAN_DIR, exist_ok=True)
 
print("Scan root folder:", SCAN_ROOT)
print("This run folder:", SCAN_DIR)
os.makedirs(SCAN_DIR, exist_ok=True) import wx