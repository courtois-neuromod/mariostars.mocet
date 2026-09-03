import os
import sys
import mocet
import time
from tqdm import tqdm
import numpy as np
from random import shuffle

from analysis.utils import resolve_paths, select_run_from_qc, parse_log, creat_timestamps_file, get_pupils_data_from_mp4

def main(source_dir_eyetracking=None, source_dir_fmriprep=None, output_dir='output_data'):
    
    qc_fname = os.path.join('source_data', 'neuromod_eyetrack_mariostars_QC.csv')
    run_list = select_run_from_qc(qc_fname)
    shuffle(run_list)
    run_list = run_list[:10]
    log_cache = []
    
    for sub, ses, run, file_nb in tqdm(run_list,desc="Processing runs",position=0):

        start_time = time.perf_counter()

        log_fname, pldata_fname, mp4_fname, confounds_fname, \
            mp4_calibration_fname, pldata_calibration_fname = resolve_paths(sub, 
                                                                            ses, 
                                                                            run, 
                                                                            file_nb, 
                                                                            source_dir_eyetracking, 
                                                                            source_dir_fmriprep
                                                                            )

        if log_fname == None:
            continue

        if (sub, ses) not in log_cache:
            log_cache.append((sub, ses))
            log_dict = parse_log(log_fname)
            # add a function to extract the number a calibration points, their positions and their onset and offset fr duration
            os.makedirs(os.path.join(output_dir, sub, ses, 'fix'), exist_ok=True)
            
        delay = log_dict[run]['delay']
        duration = log_dict[run]['duration']

        timestamps_fname = creat_timestamps_file(pldata_fname)
        _, data_fname = get_pupils_data_from_mp4(mp4_fname)

        # apply mocet
        
        pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.utils.clean_viewpoint_data(data_fname,
                                                                             timestamps_fname,
                                                                             start=delay,
                                                                             duration=duration)
        print(f'pupil_data shape: {pupil_data.shape}')

        pupil_data = mocet.apply_mocet(pupil_data, 
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence, 
                               motion_source='fmriprep',
                               polynomial_order=3)
        
        print(f'pupil_data shape: {pupil_data.shape}')

        delay_cal = 0
        duration_cal = log_dict[run]['duration_cal'] # TODO

        timestamps_fname_cal = creat_timestamps_file(pldata_calibration_fname)
        _, data_fname_cal = get_pupils_data_from_mp4(mp4_calibration_fname)

        # calibration
        pupil_data_cal, pupil_timestamps_cal, pupil_confidence_cal, _ = mocet.utils.clean_viewpoint_data(data_fname_cal,
                                                                             timestamps_fname_cal,
                                                                             start=delay_cal,
                                                                             duration=duration_cal)
        
        calibration_coordinates = log_dict[run]['coordinates'] # TODO exact from the log
        calibration_order = np.arange(len(calibration_coordinates))

        # From the log I need the calibratin coordinates and the calibration_order, checkout what it's mean exactly
        calibrator = mocet.EyetrackingCalibration(calibration_coordinates=calibration_coordinates,
                                                              calibration_order=calibration_order,
                                                              repeat=True)
        
        calibration_timestemps = log_dict[run]['timestamps'] # TODO exact from the log, start at 0

        calibration_pupils = []
        for i in calibration_order:
            start = calibration_timestemps[i][0]
            end = calibration_timestemps[i][1]
            # I need pupil timestamps relatives to the begining of the calibration
            log_effective = np.logical_and(pupil_timestamps_cal >= start * 1000, pupil_timestamps_cal < end * 1000)
            # I need the mp4 recorded during the calibration, preprocess as the data recorded during the task
            calibration_pupils.append([np.nanmean(pupil_data_cal[log_effective, 0]),
                                        np.nanmean(pupil_data_cal[log_effective, 1])])
        calibration_pupils = np.array(calibration_pupils)

        # projection
        
        calibrator.fit(calibration_pupils[:, 0], calibration_pupils[:, 1])
        gaze_coordinates = calibrator.transform(pupil_data)

        np.save(os.path.join(output_dir, sub, ses, 'fix', f'{sub}_{ses}_task-mariostars_{run}_gaze_coordinate.npy'), gaze_coordinates)
        np.save(os.path.join(output_dir, sub, ses, 'fix', f'{sub}_{ses}_task-mariostars_{run}_gaze_timestamp.npy'), pupil_timestamps)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time)/60
        print(f"Time taken for {sub}, {ses}, {run}: {execution_time:.2f} min")
       
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])