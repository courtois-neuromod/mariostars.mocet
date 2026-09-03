import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import msgpack
import eyerec

def resolve_paths(sub, ses, run, file_nb, source_dir_eyetracking, source_dir_fmriprep):
    if source_dir_eyetracking == None and source_dir_fmriprep == None:
            source_dir_eyetracking = os.path.join('source_data', 'eyetracking')
            source_dir_fmriprep = os.path.join('source_data','mariostarts.fmriprep')
    
    log_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.log')
    pldata_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'pupil.pldata')
    mp4_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'eye0.mp4')
    confounds_fname = os.path.join(source_dir_fmriprep,sub, ses, 'func', f'{sub}_{ses}_task-mariostars_run-{run[-1]}_part-mag_desc-confounds_timeseries.tsv')
    mp4_calibration_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'eyeTrackercalibration-{run}', '000', 'eye0.mp4')
    pldata_calibration_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'eyeTrackercalibration-{run}', '000', 'pupil.pldata')

    name_files = [log_fname, pldata_fname, mp4_fname, confounds_fname, mp4_calibration_fname, pldata_calibration_fname]
    if not all(os.path.isfile(f) for f in name_files):
            print(f'ERROR with not existing files: subject:{sub}, session:{ses}, file_nbfile number:{file_nb} and run:{run}')
            print('Please complet the QC file')
            print(log_fname)
            print(pldata_fname)
            print(mp4_fname)
            print(confounds_fname)
            print(mp4_calibration_fname)
            print(pldata_calibration_fname)
            return [None]*6
    
    return name_files

def select_run_from_qc(qc_fname):
    # import QC report as pd
    df_qc = pd.read_csv(qc_fname)
    
    # filter row where DO_NOT_USE!=1 & empty_log == False 
    filter_qc = ((df_qc['DO_NOT_USE']!=1)&
                 (df_qc['empty_log']==False)&
                 (df_qc['has_pupil']==True)&
                 (df_qc['has_gaze']==True)&
                  (df_qc['has_eyemovie']==True)) 
    df_filter = df_qc[filter_qc]
    
    # list the ['file_number'] not to use
    df_without_duplicates = (
        df_filter
        .groupby(['subject', 'session', 'run'])
        .first()
        .reset_index())

    # tuple of sub-ses-run to use
    df_grouped = df_without_duplicates.groupby(['subject', 'session', 'run', 'file_number'])
    run_list = []
    for keys, _ in df_grouped:
        run_list.append(keys)

    return run_list

def parse_log(log_fname):

    results = {}
    ttl_time = None
    eyetracking_start = None
    eyetracking_stop = None
    calibration_start = None
    calibration_stop = None
    calibration_timestamps = []
    coordinates = []
    run = None

    isFirst = True
    with open(log_fname, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 3:
                continue

            timestamp = float(parts[0])
            message = parts[2]

            if "starting eyetracking recording" in message and isFirst:
                calibration_start = timestamp
                isFirst = False
            elif 'calibrate_position' in message and calibration_start is not None:
                calibration_timestamps.append(timestamp)
                x_cal = int(message.split(',')[-2])
                y_cal = int(message.split(',')[-1])
                coordinates.append((x_cal, y_cal))
            elif "task - <class 'src.shared.eyetracking.EyetrackerCalibration'> :" in message and calibration_start is not None:
                calibration_stop = timestamp
            elif "fMRI TTL 0" in message and calibration_stop is not None:
                ttl_time = timestamp
            elif "starting eyetracking recording" in message and ttl_time is not None:
                eyetracking_start = timestamp
            elif "stopping eyetracking recording" in message and eyetracking_start is not None:
                eyetracking_stop = timestamp
            elif "task - <class 'src.tasks.videogame.VideoGameMultiLevel'> :" in message and eyetracking_stop is not None:
                run = message.split('_')[1][0:6]

            if run is not None:
                delay = eyetracking_start - ttl_time
                duration = eyetracking_stop - eyetracking_start
                duration_cal = calibration_stop - calibration_start

                first_timestamp = calibration_timestamps[0]
                calibration_intervals = [
                    (start - first_timestamp, start - first_timestamp + 3)
                    for start in calibration_timestamps
                ]
                
                results[run] = {'delay': delay,
                                'duration': duration,
                                'duration_cal': duration_cal,
                                'coordinates':coordinates,
                                'timestamps':calibration_intervals
                                }
                
                ttl_time = None
                eyetracking_start = None
                eyetracking_stop = None
                run = None
                isFirst = True

    return results

# little game plan for parse log:
#   il fuat le 'starting eyetracking recording' pui la 'eyetracker_calibration: starting' pour prendre le début de la calib

def extract_timestamps(pldata_fname):
    timestamps = []
    with open(pldata_fname, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for topic, payload in unpacker:
            packet = msgpack.unpackb(payload, raw=False)
            timestamps.append(packet['timestamp'])
    return timestamps

def creat_timestamps_file(pldata_fname):
    timestamps = np.array(extract_timestamps(pldata_fname))
    pupil_onset_deltas = np.diff(timestamps, prepend=timestamps[0])
    print(f"deltas shape: {pupil_onset_deltas.shape}")
    timestamps_fname = 'recording-eyetracking_timestamps.txt'
    if os.path.exists(timestamps_fname):
        os.remove(timestamps_fname)
    with open(timestamps_fname, "a") as file:
        for delta in pupil_onset_deltas:
            file.write(f"10\ttotal_time\t{delta*1000}\n")
            file.write("777\ttotal_time\tMovieFrame\n")
    return timestamps_fname

def get_pupils_data_from_mp4(mp4_fname):
    capture = cv2.VideoCapture(mp4_fname)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    read, frame = capture.read()

    tracker = eyerec.PupilTracker(name='purest')

    df = pd.DataFrame(columns=['diameter_px','width_px','height_px','axisRatio',
                                'center_x','center_y', 'angle_deg', 'confidence'])
    count = 0
    with tqdm(total=total_frames, desc="Detecting pupils", unit="frame", leave=False) as pbar:
        while read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pupil = tracker.detect(capture.get(cv2.CAP_PROP_POS_MSEC), frame)
            
            df.loc[count] = {'diameter_px':np.max([pupil['size']]),
                                'width_px':pupil['size'][0],
                                'height_px':pupil['size'][1],
                                'axisRatio':np.min([pupil['size']])/np.max([pupil['size']]),
                                'center_x':pupil['center'][0],
                                'center_y':pupil['center'][1],
                                'angle_deg': pupil['angle'],
                                'confidence':pupil['confidence']}
                
            read, frame = capture.read()
            count += 1
            pbar.update(1)
    capture.release()
    data_fname = f'recording-eyetracking_physio_log.csv'
    df.to_csv(data_fname, index=False)
    print(f"pupil df shape: {df.shape}")
    return df, data_fname

def capture_calibration(log_fname):
    calibration_timestamps = []
    calibration_positions = []
    return calibration_timestamps, calibration_positions