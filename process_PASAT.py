#!/usr/bin/env/python
import numpy as np
from pydub import silence, AudioSegment
import csv
import soundfile
import speech_recognition as sr
import os
from math import isnan

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TUNABLE PARAMETERS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Trial name (subject name, etc)
TRIAL_NAME = "pasat_test"
CSV_FILENAME = TRIAL_NAME + ".csv"
# Pause time in seconds
DELAY = 2.0
# Number of tests (Max 60)
NUM_TESTS = 30
# NUM_TESTS = 60
# The highest audio level (in dB) the program will determine to be considered "silence"
SILENCE_THRESHOLD_DB = -20.0
# The minimum period, in milliseconds, that could distinguish two different responses
MIN_PERIOD_SILENCE_MS = 100

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
WORD_TO_NUM = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
               "TEN": 10}


# Normalize audio file to given target dB level - https://stackoverflow.com/questions/59102171/getting-timestamps-from-audio-using-pythons
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == "__main__":
    # Get relevant data from csv file
    file = open(CSV_FILENAME)
    reader = csv.reader(file)
    header = next(reader)
    data = []
    for row in reader:
        if len(row) > 0:
            data.append(row)
    data = np.array(data)
    # Reconstruct original number array from the two given from the csv (one is just offset by one number of the other)
    number_array = np.array(data[:, 0], dtype=int)
    number_array_step_inc = np.array(data[:, 1], dtype=int)
    number_array = np.append(number_array, number_array_step_inc[-1])
    answer_array = np.array(data[:, 2], dtype=int)
    stimuli_time_stamps = np.array(data[:, 3], dtype=float)
    NUM_TESTS = stimuli_time_stamps.size

    print("Interpret data (this may take a while)...")
    # Open .wav with pydub
    audio_segment = AudioSegment.from_wav(TRIAL_NAME + ".wav")
    rec_seconds = audio_segment.duration_seconds

    # Normalize audio_segment to a threshold
    normalized_sound = match_target_amplitude(audio_segment, SILENCE_THRESHOLD_DB)

    # Generate nonsilent chunks (start, end) with pydub
    response_timing_chunks = np.array(
        silence.detect_nonsilent(normalized_sound, min_silence_len=MIN_PERIOD_SILENCE_MS,
                                 silence_thresh=SILENCE_THRESHOLD_DB,
                                 seek_step=1))

    # If unable to detect nonsilence, end program and notify user
    if len(response_timing_chunks) == 0:
        print("Could not detect user's responses. Silence threshold/Minimum silence period may need tuning.")
        exit(1)

    # Calculate the time that the user starts to speak in each nonsilent "chunk"
    response_timing_markers = np.array(response_timing_chunks[:, 0]) / 1000.0
    while response_timing_markers[0] == 0.0:
        response_timing_markers = np.delete(response_timing_markers, 0)
        response_timing_chunks = np.delete(response_timing_chunks, 0, 0)

    # Create a folder to store the individual responses as clips to help determine
    # response accuracies later on.
    clip_seperation_path = TRIAL_NAME + "_reponse_chunks"
    if not os.path.isdir(clip_seperation_path):
        os.mkdir(clip_seperation_path)
    # How much we add (ms) to the ends of a clip when saved
    clip_threshold = 600
    for i in range(len(response_timing_chunks)):
        chunk = response_timing_chunks[i]
        chunk_filename = os.path.join(clip_seperation_path, f"chunk{i}.wav")
        # Save the chunk as a serperate wav, acounting for the fact it could be at the very beggining or end
        if chunk[0] <= clip_threshold:
            (audio_segment[0:chunk[1] + clip_threshold]).export(chunk_filename, format="wav")
        elif chunk[1] >= ((rec_seconds * 1000.0) - clip_threshold - 1):
            (audio_segment[chunk[0] - clip_threshold:(rec_seconds * 1000) - 1]).export(chunk_filename, format="wav")
        else:
            (audio_segment[chunk[0] - clip_threshold:chunk[1] + clip_threshold]).export(chunk_filename,
                                                                                        format="wav")
        # Reformat the wav files using soundfile to allow for speech recongition, and store in folder
        data, samplerate = soundfile.read(chunk_filename)
        soundfile.write(chunk_filename, data, samplerate, subtype='PCM_16')

    # Create an array to hold users response accuracy (TRUE, FALSE, or N/A)
    response_accuracies = []

    # Init the speech to text recognizer
    r = sr.Recognizer()

    # Create an array to hold raw user responses
    raw_responses = []

    # Calculate the reponse times given the arrays for response_timing_markers and stimuli_time_stamps
    reaction_times = []
    clip_index_array = np.empty(NUM_TESTS, dtype=int)
    num_correct_responses = 0
    for i in range(NUM_TESTS):
        # If there is no response after a time stamp, clearly the user failed to respond...
        rt = float('nan')
        clip_index_array[i] = -9999
        if stimuli_time_stamps[i] > response_timing_markers[-1]:
            response_accuracies.append("N/A")
            raw_responses.append("N/A")
            reaction_times.append(float('nan'))
        else:
            # Determine the most accurate nonsilent chunk that is associated with a given iteration
            for j in range(len(response_timing_markers)):
                if response_timing_markers[j] > stimuli_time_stamps[i]:
                    # If reaction is too fast, it means the program is considering a delayed response from previous stimulus
                    # Thus, we should continue the loop if that is the case, otherwise, break and store the reaction time
                    if response_timing_markers[j] - stimuli_time_stamps[i] < 0.1 and len(reaction_times) > 0 and \
                            reaction_times[-1] > DELAY:
                        continue
                    rt = response_timing_markers[j] - stimuli_time_stamps[i]
                    break
            # If there is no nonsilent chunk after the time that the stimulus is displayed, store reaction time as "nan"
            # Also if the user's response is over 1.6s after the stimulus is displayed, then we know they either failed to
            # respond or the audio was not recorded and intepreted properly.
            if j >= len(response_timing_markers) or (rt > DELAY * 1.5):
                reaction_times.append(float('nan'))
                raw_responses.append("N/A")
                response_accuracies.append("N/A")
                continue
            else:
                # Save index to clip index array
                clip_index_array[i] = j
                # If the response was valid, detemine if it was correct using speech recognition
                with sr.AudioFile(os.path.join(clip_seperation_path, f"chunk{j}.wav")) as source:
                    # listen for the data (load audio to memory)
                    audio_data = r.record(source)
                    # recognize (convert from speech to text)
                    resp = "Undetected"
                    resp_backup = "Undetected"
                    try:
                        resp = (r.recognize_google(audio_data).split()[0])
                        if isinstance(resp, str):
                            resp = resp.upper()
                    except sr.UnknownValueError as err1:
                        # recognize (convert from speech to text) with sphinx instead of google
                        try:
                            resp_backup = (r.recognize_sphinx(audio_data).split()[0]).upper()
                            if isinstance(resp_backup, str):
                                resp_backup = resp_backup.upper()
                        # If no response can be determined, report accuracies as N/A, store reaction time, and move on
                        except sr.UnknownValueError as err2:
                            response_accuracies.append("N/A")
                            raw_responses.append("N/A")
                            reaction_times.append(rt)
                            continue
                    if (resp.isnumeric() and answer_array[i] == int(resp)) or (resp in WORD_TO_NUM.keys() and WORD_TO_NUM[resp] == answer_array[i]):
                        response_accuracies.append("TRUE")
                        num_correct_responses += 1
                        raw_responses.append(resp)
                    elif (resp_backup.isnumeric() and answer_array[i] == int(resp_backup)) or (resp_backup in WORD_TO_NUM.keys() and WORD_TO_NUM[resp_backup] == answer_array[i]):
                        response_accuracies.append("TRUE")
                        num_correct_responses += 1
                        raw_responses.append(resp_backup)
                    # If word not found, store response and mark as false
                    else:
                        raw_responses.append(resp)
                        response_accuracies.append("FALSE")
            reaction_times.append(rt)

    # Create another array to label each reactiontime according to if it was within the alloted time or not
    reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
    for i in range(NUM_TESTS):
        if reaction_times[i] > DELAY or isnan(reaction_times[i]):
            reaction_on_time[i] = False
        else:
            reaction_on_time[i] = True

    # Write results to file
    with open(TRIAL_NAME + "_RESULTS.csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['1st number', '2nd number', 'User response', 'Correct answer', 'Accuracy (T/F)', 'Reaction time (s)',
             'Reaction on time (T/F)', 'Clip Index', ' ', ' ', 'Time (from start) of responses'])
        num_rows_in_table = max([len(response_timing_markers), len(answer_array)])
        for i in range(num_rows_in_table):
            if i >= len(response_timing_markers):
                writer.writerow([number_array[i], number_array[i + 1], raw_responses[i], answer_array[i],
                                 response_accuracies[i], reaction_times[i], reaction_on_time[i], clip_index_array[i], ' ', ' ', -1.0])
            elif i >= len(answer_array):
                writer.writerow([-1, -1, -1, -1, -1, -1, -1, -1, ' ', ' ', response_timing_markers[i]])
            else:
                writer.writerow([number_array[i], number_array[i + 1], raw_responses[i], answer_array[i],
                                 response_accuracies[i], reaction_times[i], reaction_on_time[i], clip_index_array[i], ' ', ' ', response_timing_markers[i]])
    print("Done.")
