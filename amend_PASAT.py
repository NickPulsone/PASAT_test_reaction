import numpy as np
from pydub import silence, AudioSegment
import csv
import soundfile
import speech_recognition as sr
import os
from math import isnan

# Clip indices of clips to discard
REMOVE_CLIPS = [10]

# Word to number dictionary
WORD_TO_NUM = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
               "TEN": 10}

# Pause time in seconds
DELAY = 2.0

# Trial name and name of csv file containing existing results to be modified
TRIAL_NAME = "pasat_test1"
TRIAL_CSV_FILENAME = TRIAL_NAME + ".csv"
RESULTS_CSV_FILENAME = TRIAL_NAME + "_results.csv"
CLIP_SEPERATION_PATH = TRIAL_NAME + "_reponse_chunks"
CHUNK_DIR_NAME = "pasat_test1_reponse_chunks"

# Get data from trial csv file
trial_file = open(TRIAL_CSV_FILENAME)
trial_reader = csv.reader(trial_file)
trial_header = next(trial_reader)
data = []
for row in trial_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from trial csv file
stimuli_time_stamps = np.array(data[:, 3], dtype=float)

# Get data from results csv file
results_file = open(RESULTS_CSV_FILENAME)
results_reader = csv.reader(results_file)
results_header = next(results_reader)
data = []
for row in results_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from results csv file
number_1_array = np.array(data[:, 0], dtype=int)
number_2_array = np.array(data[:, 1], dtype=int)
user_responses = np.array(data[:, 2], dtype=str)
response_timing_markers = np.array(data[:, 3], dtype=float)
correct_answers = np.array(data[:, 4], dtype=int)
accuracy_array = np.array(data[:, 5], dtype=str)
reaction_times = np.array(data[:, 6], dtype=float)
reaction_on_time = np.array(data[:, 7], dtype=str)
clip_index_array = np.array(data[:, 8], dtype=int)
NUM_TESTS = correct_answers.size

# Get the number of clips by counting the nubmer of clips in the folder
total_num_clips = 0
dir = CHUNK_DIR_NAME
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        total_num_clips += 1

# Get index of the iteration of each corresponding clip in question
num_remove_clips = len(REMOVE_CLIPS)
iteration_indices = np.empty(num_remove_clips, dtype=int)
for i in range(num_remove_clips):
    iteration_indices[i] = np.where(clip_index_array == REMOVE_CLIPS[i])[0][0]
clip_iteration_range = tuple(i for i in range(total_num_clips) if i not in REMOVE_CLIPS)
print(clip_iteration_range)

# Redo the analysis for the iterations ignoring the given remove indices
# Init the speech to text recognizer
r = sr.Recognizer()
for i in iteration_indices:
    # If there is no response after a time stamp, clearly the user failed to respond...
    rt = float('nan')
    clip_index_array[i] = -1
    if stimuli_time_stamps[i] > response_timing_markers[-1]:
        accuracy_array[i] = "N/A"
        user_responses[i] = "N/A"
    else:
        # Determine the most accurate nonsilent chunk that is associated with a given iteration
        for j in clip_iteration_range:
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
            reaction_times[i] = float('nan')
            user_responses[i] = "N/A"
            accuracy_array[i] = "N/A"
            continue
        else:
            # Save index to clip index array
            clip_index_array[i] = j
            # If the response was valid, detemine if it was correct using speech recognition
            with sr.AudioFile(os.path.join(CLIP_SEPERATION_PATH, f"chunk{j}.wav")) as source:
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
                        accuracy_array[i] = "N/A"
                        user_responses[i] = "N/A"
                        reaction_times[i] = rt
                        continue
                if (resp.isnumeric() and correct_answers[i] == int(resp)) or (resp in WORD_TO_NUM.keys() and WORD_TO_NUM[resp] == correct_answers[i]):
                    accuracy_array[i] = "TRUE"
                    user_responses[i] = resp
                elif (resp_backup.isnumeric() and correct_answers[i] == int(resp_backup)) or (resp_backup in WORD_TO_NUM.keys() and WORD_TO_NUM[resp_backup] == correct_answers[i]):
                    accuracy_array[i] = "TRUE"
                    user_responses[i] = resp_backup
                # If word not found, store response and mark as false
                else:
                    user_responses[i] = resp
                    accuracy_array[i] = "FALSE"
        reaction_times[i] = rt

# Create another array to label each reactiontime according to if it was within the alloted time or not
reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
for i in iteration_indices:
    if reaction_times[i] > DELAY or isnan(reaction_times[i]):
        reaction_on_time[i] = False
    else:
        reaction_on_time[i] = True

# overwrite results file
with open(TRIAL_NAME + "_RESULTS.csv", 'w') as reac_file:
    writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['1st number', '2nd number', 'User response', 'Time (from start) of user response', 'Correct answer', 'Accuracy (T/F)', 'Reaction time (s)',
         'Reaction on time (T/F)', 'Clip Index'])
    for i in range(NUM_TESTS):
        writer.writerow([number_1_array[i], number_2_array[i], user_responses[i], response_timing_markers[i], correct_answers[i],
                         accuracy_array[i], reaction_times[i], reaction_on_time[i], clip_index_array[i]])
print("Done.")
