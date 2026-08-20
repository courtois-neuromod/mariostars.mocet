import os
import sys
import mocet
import time

# IMPORT FOR THE FUNCTIONS THAT WILL BE PLACED IN UTILS
import cv2
import eyerec
import numpy as np
import pandas as pd
import msgpack


# FUTURE IMPORT
# from analysis import select_run_from_qc, find_time_between_rec_and_start, creat_data_file, get_pupils_data_from_mp4


def select_run_from_qc(qc_fname):
    # import QC report as pd
    df_qc = pd.read_csv(qc_fname)
    
    # filter row where DO_NOT_USE!=1 & empty_log == False 
    filter_qc = ((df_qc['DO_NOT_USE']!=1)&(df_qc['empty_log']==False)) 
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

def find_delays_and_durations(log_fname):

    results = {}
    ttl_time = None
    eyetracking_start = None
    eyetracking_stop = None
    run = None

    with open(log_fname, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 3:
                continue

            timestamp = float(parts[0])
            message = parts[2]

            if "fMRI TTL 0" in message:
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
                results[run] = {'delay': delay, 'duration': duration}
                
                ttl_time = None
                eyetracking_start = None
                eyetracking_stop = None
                run = None

    return results


def read_pldata(pldata_fname):
    packets = []
    with open(pldata_fname, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for topic, payload in unpacker:
            packet = msgpack.unpackb(payload, raw=False)
            packets.append((topic, packet))
    return packets

def extract_timestamps(pldata_fname):
    timestamps = []
    with open(pldata_fname, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for topic, payload in unpacker:
            packet = msgpack.unpackb(payload, raw=False)
            timestamps.append(packet['timestamp'])
    return timestamps

def creat_data_file(pldata_fname):
    timestamps = np.array(extract_timestamps(pldata_fname))
    pupil_onset_deltas = np.diff(timestamps, prepend=timestamps[0])
    timestamps_fname = 'recording-eyetracking_timestamps.txt'
    with open(timestamps_fname, "a") as file:
        for delta in pupil_onset_deltas:
            file.write(f"10\ttotal_time\t{delta}16.6679\n")
            file.write("777\ttotal_time\tMovieFrame\n")
    return timestamps_fname


def get_pupils_data_from_mp4(mp4_fname):
    capture = cv2.VideoCapture(mp4_fname)
    read, frame = capture.read()

    tracker = eyerec.PupilTracker(name='purest')

    df = pd.DataFrame(columns=['diameter_px','width_px','height_px','axisRatio',
                                'center_x','center_y', 'angle_deg', 'confidence'])
    count = 0
    while True:
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
        if not read:
            break
        count += 1
    data_fname = f'recording-eyetracking_physio_log.csv'
    df.to_csv(data_fname, index=False)
    return df, data_fname

def main(source_data_eyetracking='', source_data_fmriprep=''):
    
    qc_fname = os.path.join('source_data', 'neuromod_eyetrack_mariostars_QC.csv')

    run_list = select_run_from_qc(qc_fname)
    
    for sub, ses, run, file_nb in run_list:

        start_time = time.perf_counter()

        if source_data_eyetracking == '' and source_data_fmriprep == '':
            log_fname = os.path.join('source_data', 'eyetracking', sub, ses, f'{sub}_{ses}_{file_nb}.log')
            pldata_fname = os.path.join('source_data', 'eyetracking',sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'pupil.pldata')
            mp4_fname = os.path.join('source_data','eyetracking', sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'eye0.mp4')
            confounds_fname = os.path.join('source_data','mariostarts.fmriprep',f'{sub}_{ses}_task_{run}_desc-confounds_timeseries.tsv')
        else:
            log_fname = os.path.join(source_data_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.log')
            pldata_fname = os.path.join(source_data_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'pupil.pldata')
            mp4_fname = os.path.join(source_data_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'eye0.mp4')
            confounds_fname = os.path.join(source_data_fmriprep,sub, ses, 'func', f'{sub}_{ses}_task-mariostars_run-{run[-1]}_part-mag_desc-confounds_timeseries.tsv')

        delays_durations = None

        if not all(os.path.isfile(f) for f in [log_fname, pldata_fname, mp4_fname, confounds_fname]):
            print(f'ERROR with not existing files: subject:{sub}, session:{ses}, file_nbfile number:{file_nb} and run:{run}')
            print('Please complet the QC file')
            #print(log_fname)
            #print(pldata_fname)
            #print(mp4_fname)
            #print(confounds_fname)
            continue
        
        if delays_durations == None:
            delays_durations = find_delays_and_durations(log_fname)
        
        
        delay = delays_durations[run]['delay']
        duration = delays_durations[run]['duration']

        timestamps_fname = creat_data_file(pldata_fname)
        _, data_fname = get_pupils_data_from_mp4(mp4_fname)

        pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.utils.clean_viewpoint_data(data_fname,
                                                                             timestamps_fname,
                                                                             start=delay,
                                                                             duration=duration)

        pupil_data = mocet.apply_mocet(pupil_data, 
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence, 
                               motion_source='fmriprep',
                               polynomial_order=3)

        # save output

        print(pupil_data)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Time taken for {sub}, {ses}, {run}: {execution_time:.6f} seconds")
        break

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
