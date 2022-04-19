#!/usr/bin/env/python
import numpy as np
from scipy.io import loadmat, wavfile
from time import sleep, time
import pyaudio
import sounddevice as sd
import datetime
import csv

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TUNABLE PARAMETERS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Trial name (subject name, etc)
TRIAL_NAME = "pasat_test_419_1"
# Name of the test sequence file
TEST_QUESTION_FILENAME = "PASAT_versionA_HO.mat"
# Pause time in seconds
DELAY = 2.0
# Number of tests (Max 60)
NUM_TESTS = 40
# NUM_TESTS = 60
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
WORD_TO_NUM = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
               "TEN": 10}

if __name__ == "__main__":
    """
    # Initialize engine for TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    """

    # Load sound and frequency data
    mat = loadmat(TEST_QUESTION_FILENAME)
    number_array = (mat["ind"])[::, 0]
    answer_array = (mat["answer"])[::, 0]
    Number = loadmat("Number.mat")

    # Initialize array to contain time data
    stimuli_time_stamps = np.empty(NUM_TESTS, dtype=datetime.datetime)

    # Give user a countdown before recording is started
    print("Get ready...")
    for num in ["3..", "2..", "1.."]:
        print(num)
        sleep(1)
    print("Starting the recording...")

    # Define recording parameters and start recording
    rec_seconds = int(NUM_TESTS) * (DELAY + 1.7)
    rec_sample_rate = 44100
    myrecording = sd.rec(int(rec_seconds * rec_sample_rate), samplerate=rec_sample_rate, channels=1)
    recording_start_time = datetime.datetime.now()
    sleep(DELAY)

    # Open a data stream to play audio
    p = pyaudio.PyAudio()
    fs = Number["Fs" + str(number_array[0])][0][0]
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

    # Display first stimulus info
    print(f"Initial audio: Number={number_array[0]}, Answer=N/A")

    # Play the first sound and pause
    number_sound = (Number["y" + str(number_array[0])])[:, 0]
    stream.write(number_sound.astype(np.float32).tobytes())
    htime = time()
    while (time() - htime) < DELAY:
        sleep(0.01)

    # Run the tests, playing the rest of the sounds
    for i in range(1, (NUM_TESTS + 1)):
        # Display current stimulus info
        print(f"Stimulus {i}: Number={number_array[i]}, Answer={answer_array[i-1]}")
        # Record the time, play the sound
        number_sound = (Number["y" + str(number_array[i])])[:, 0]
        stimuli_time_stamps[i - 1] = datetime.datetime.now()
        stream.write(number_sound.astype(np.float32).tobytes())
        htime = time()
        # Pause
        while (time() - htime) < DELAY:
            sleep(0.01)

    # Close audio data stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Stop the main recording, save file as .wav
    print("Waiting for recording to stop...")
    sd.wait()
    wavfile.write(TRIAL_NAME + '.wav', rec_sample_rate, myrecording)
    print("Done. Saving data...")
    # Calculate the time of each stimulus with respect to the start of the recording
    stimuli_time_stamps = np.array(
        [(stimuli_time_stamps[i] - recording_start_time).total_seconds() for i in range(NUM_TESTS)])

    # Write results to file
    with open(TRIAL_NAME + ".csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['1st number', '2nd number', 'Correct answer', 'Stimuli time from start (s)'])
        for i in range(NUM_TESTS):
            writer.writerow([number_array[i], number_array[i + 1], answer_array[i], stimuli_time_stamps[i]])
    print("Done.")
