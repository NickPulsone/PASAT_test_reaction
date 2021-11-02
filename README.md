# PASAT_test_reaction
Implements a psychological test where a user is prompted a series of integers (1-5). The user must respond after each of number starting from the second one prompted, with the sum of the number that was just said with the previous number.
For example if the user was prompted with [3, 1, 4, 5, 2]
The correct response would be [4, 5, 9, 7]

Requires Python 3.9. Edit tunable paramaters as necessary in "PASAT.py."

IMPORTANT: Include the files in this drive link in your working directory (too big for github): https://drive.google.com/drive/folders/1_XCEDEXR9AgY9L-dRdYDVTmz9gXPXfcK?usp=sharing

If the program is unable to calculate the reaction time of a given response (whether it be the because the user failed to respond, the microphone did not pick up user audio, or otherwise) the reaction time will be recorded as "nan."
