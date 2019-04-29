# q_learning_dots_and_boxes
This project is an implementation of a Q learning algorithm which trains itself through self-play to play the game of dots and boxes. The q learned player is then tested against a random move player.

## Instructions
* First run the train_q_table file: `python train_q_table.py`
* Then run the test_q_table file: `python test_q_table.py` to view scores graph over 5000 games
* Or run the play_3_3 file: `python play_3_3.py` to view one game live

In order to run the 2x2 version of the game change BOARD_SIZE = 3 in the train and test files
In order to run the 3x3 version of the game change BOARD_SIZE = 4 in the train and test files

Then repeat the above steps
