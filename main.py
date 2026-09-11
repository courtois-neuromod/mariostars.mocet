import os
import sys
import mocet
import time
from tqdm import tqdm
import numpy as np
from random import shuffle

from analysis.utils import resolve_paths, select_run_from_qc, extract_data, extract_calibration_data

def main(source_dir_eyetracking=None, source_dir_fmriprep=None):
    
    qc_fname = os.path.join('source_data', 'neuromod_eyetrack_mariostars_QC.csv')
    run_list = select_run_from_qc(qc_fname)
    #shuffle(run_list)
    
    for sub, ses, run, file_nb in tqdm(run_list,desc="Processing runs",position=0):

        start_time = time.perf_counter()

        pldata_fname, confounds_fname, calibration_data_fname = resolve_paths(sub, 
                                                                            ses, 
                                                                            run, 
                                                                            file_nb, 
                                                                            source_dir_eyetracking, 
                                                                            source_dir_fmriprep
                                                                            )
        if pldata_fname == None:
            continue

        pupil_data, pupil_timestamps, pupil_confidence, _ = extract_data(pldata_fname)

        if pupil_data.shape[0] != pupil_timestamps.shape[0]:
            print(f'Passing {sub}_{ses}_{run}_{file_nb} because the number of timestamp doesn\'t match the number of pupil observed')
            continue

        pupil_data = mocet.apply_mocet(pupil_data,
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence, 
                               motion_source='fmriprep',
                               polynomial_order=3)

        # calibration
        markers_pos, markers_order, pupil_mean_pos = extract_calibration_data(calibration_data_fname)

        if markers_pos.size == 0:
            continue
        
        calibrator = mocet.EyetrackingCalibration(calibration_coordinates=markers_pos,
                                                              calibration_order=markers_order,
                                                              repeat=False)
   
        calibrator.fit(pupil_mean_pos[:, 0], pupil_mean_pos[:, 1])
        gaze_coordinates = calibrator.transform(pupil_data)

        output_dir = os.path.join('output_data', sub, ses, 'eyetracking')
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'{sub}_{ses}_task-mariostars_{run}_gaze_coordinate.npy'), gaze_coordinates)
        np.save(os.path.join(output_dir, f'{sub}_{ses}_task-mariostars_{run}_gaze_timestamp.npy'), pupil_timestamps)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time)
        print(f"Time taken for {sub}, {ses}, {run}: {execution_time:.2f}")
       
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])