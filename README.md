# M2_automized
This set of codes allows to run and analyze the M2 of a mid-IR beam. Must connect the step motor (from PI), and the WinCam D. The camera must be placed on the step motor. You should run them with this order:
1. acquire_frame
2. create_folder
3. initialize_camera
4. scan_stage
5. analyze_scan
Afterwards, to take another measurement you must just run 2. again (create another folder), and 4. (scan_stage). File 5. (analyze_scan) analyzes the last folder created. 
