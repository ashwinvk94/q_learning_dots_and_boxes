#!/usr/bin/env python
'''
ENPM 673 Spring 2019: Robot Perception
Project 3 GMM Bouy Detection

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)
Graduate Students in Robotics,
University of Maryland, College Park

Acknowledgment:  Stephen Roller (https://gist.github.com/stephenroller)

'''
import curses, sys, re
from time import sleep
from random import randint
import numpy as np
import pickle
import matplotlib.pyplot as plt

OFFSET_X, OFFSET_Y = 2, 1
BOARD_SIZE = 4
#

# ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

enable_viz = 1
np.set_printoptions(threshold=sys.maxsize)
if enable_viz:
	# init curses
	stdscr = curses.initscr()
	curses.noecho()
	curses.start_color()

	# stdscr.keypad(1)

	curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
	curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_RED)

def init_board():
	# The structure of the board is a bit confusing, but it looks like this
	# The even indexed rows have fewer values because there are fewer possible
	# lines on them.

	# o---o   o    [  [ 1, 0 ],
	# |               [ 1, 0, 0], 
	# o   o---o       [ 0, 1 ],
	#     |           [ 0, 1, 0],
	# o---o---o       [ 1, 1]           ]

	#The board is divided into rows which are represnted as arrays
	#The even rows correspond to the horizontal lines
	#The odd rows correspond to the vertical lines
	
	even = [0 for i in xrange(BOARD_SIZE - 1)]
	odd = [0 for i in xrange(BOARD_SIZE)]
	
	board = []
	for i in xrange(BOARD_SIZE * 2 - 1):
		if i % 2 == 0:
			board.append(even[:])
		else:
			board.append(odd[:])
	return board

def init_score():
	# The scorecard is a little bit redundant, but makes things easier
	# Its size is (BOARD_SIZE-1)^2 because we have to store more lines than
	# dots.
	#
	# o - o   o       [  
	# | 1 |              [ 1, None ],
	# o - o - o       
	#     | 2 |          [ None, 2 ]
	# o   o - o                       ]
	#

	# The score card keep tracks of which box belongs to which player

	return [[None for a in xrange(BOARD_SIZE-1)] for b in xrange(BOARD_SIZE-1)]

def init_q_table(nBoard):
	# This function initializes the q table contains all possible states 
	# and all possible moves at each state. The positions not possible are
	# marked specified as np.inf

	q_table = {}

	# nBoard = len(state)
	nStates = 2**nBoard
	
	return q_table,nStates

# The input is a list of current state.ie a flattened board
def find_possible_actions(state,board_size):
	all_board_indices = range(board_size)
	zero_board_indices = []
	for k in all_board_indices:
		if not state[k]:
			zero_board_indices.append(k)

	return zero_board_indices

def flatten_board(board):
	state = []
	for row in board:
		for element in row:
			state.append(element)
	return state

def create_state_to_board_mapping():
	mapping = []
	for row in xrange(BOARD_SIZE * 2 - 1):
		if row % 2 == 0:
			for col in xrange(BOARD_SIZE - 1):
				mapping.append([row,col])
		else:
			for col in xrange(BOARD_SIZE):
				mapping.append([row,col])
	return mapping

def calc_all_states(state):
	all_states = []
	n = len(state)
	nBoard = len(state)
	for i in range(2**nBoard):
		print(i)
		print(2**nBoard)
		b = bin(i)[2:]
		l = len(b)
		b = str(0) * (n - l) + b
		all_states.append([int(d) for d in b])
	return all_states

def closest_free(board, x, y):
	def distance(x1, y1, x2, y2):
		from math import sqrt
		return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	
	distances = []
	for i,row in enumerate(board):
		for j,val in enumerate(row):
			if val == 0: 
				distances.append((distance(x, y, i, j), i, j))
	
	if len(distances) == 0:
		return 0
	
	distances.sort(key=lambda x: x[0])
	return distances[0][1:]
	
def first_available_move(board):
	try:
		return closest_free(board, 0, 0)
	except IndexError:
		return 0

def check_board_full(board):
	for row in board:
		for element in row:
			if not element:
				return 0
	return 1

def generateRandxy(board):
	x = randint(0,BOARD_SIZE * 2 - 2)
	if x%2==0:
		y = randint(0,BOARD_SIZE-2)
	else:
		y = randint(0,BOARD_SIZE-1)
	return x, y

def draw_dot(x, y):
	sx = OFFSET_X + x * 4
	sy = OFFSET_Y + y
	stdscr.addstr(sy, sx, "o")
	stdscr.addstr(sy, sx + 4, "o")

def draw_line(x, y, color=0):
	# print "%d, %d" % (x, y)
	if y % 2 == 0:
		sy = OFFSET_Y + y
		sx = OFFSET_X + x * 4 + 1
		stdscr.addstr(sy, sx, '---', curses.color_pair(color))
	else:
		sy = OFFSET_Y + y
		sx = OFFSET_X + x * 4
		stdscr.addstr(sy, sx, "|", curses.color_pair(color))

def draw_filling(x, y, player):
	screen_x = OFFSET_X + x * 4 + 2
	screen_y = OFFSET_Y + y * 2 + 1
	stdscr.addstr(screen_y, screen_x, str(player), curses.color_pair(player))

def find_max_index(q_table,state, possible_indices,epsilonScaled):
	# print 'q table'
	# print q_table[state_index,:]
	# print 'a table choose'
	# max_weight_index = np.argmax(q_table[state_index,:])
	# print np.random.rand(currStateWeight.shape[0])

	currStateWeight = np.array(q_table[str(state)])
	randWeights = np.random.rand(currStateWeight.shape[0])
	updatedCurrStateWeight = epsilonScaled*randWeights+(1-epsilonScaled)*currStateWeight
	for index in range(updatedCurrStateWeight.shape[0]):
		if np.isnan(updatedCurrStateWeight[index]):
			updatedCurrStateWeight[index] = -np.inf
	# print currStateWeight
	# print randWeights
	# print updatedCurrStateWeight
	# quit()
	max_weight = np.amax(updatedCurrStateWeight)
	# print max_weight
	max_indices = np.where(updatedCurrStateWeight==max_weight)
	# np.random.rand(len(board_flatten))
	max_indices = max_indices[0]
	# print max_indices
	if len(max_indices)!=1:
		randMaxInd = randint(0,len(max_indices)-1)
		# print 'len(max_indices)!=1'
		# print randMaxInd
		max_index = max_indices[randMaxInd]
	else:
		# print 'only one'
		max_index = max_indices[0]
		# uit()
	# print q_table[state_index,:]
	# print max_index
	# raw_input('wait')
	# print max_weight_index
	# print 'asdfasd'
	# print max_weight_index
	# We are choosing the first max value 
	return max_index

def reward_check(filled_boxes,filled,drawFlag):
	if filled and not drawFlag:
		reward = 5
	elif filled and drawFlag:
		reward = 0
	elif filled_boxes>0:
		reward = 1
	else:
		reward = 0
		

	return reward

def update_q_table(qtable,prev_action_index,prev_state,curr_state,reward,learning_rate,discount_factor,all_board_indices):
	if str(curr_state) in qtable:
		# print qtable[curr_state,:]
		max_action = np.amax(qtable[str(curr_state)])
	else:
		max_action = 0

	if str(prev_state) not in qtable:
		actions = []
		for k in all_board_indices:
			if prev_state[k]==1:
				actions.append(-np.inf)
			else:
				actions.append(0)
		qtable.update({str(prev_state) : actions})
	# print 'previouis board state'
	# print prev_board
	# print 'reward'
	# print reward
	# print 'previous state'
	# print qtable[prev_state,:]
	# print 'current state'
	# print qtable[curr_state_index,:]
	# print 'max value of current state'
	# print 'max_action'
	# print max_action
	# qtable["iphone 5S"]\
	actions = qtable[str(prev_state)]
	temp = actions[prev_action_index]
	actions[prev_action_index] +=  learning_rate*(reward+discount_factor*max_action-temp)
	qtable[str(prev_state)] = actions
	# print 'previous state after update'
	# print qtable[prev_state_index,:]
	# raw_input('click enter')
	return qtable


def get_final_score(scores):
	player1, player2 = 0, 0
	for i, row in enumerate(scores):
		for j, score in enumerate(row):
			if score == 1: player1 += 1
			if score == 2: player2 += 1
	return player1,player2

def getRandMove(board):
	# print 'executing random move'
	# Select random move by checking if each randomly generated one is filled
	valid_move = 0
	random_selections = []
	# for row in board:
	# 	if len(row)==BOARD_SIZE:
	# 		print row
	# 	else:
	# 		print '   ',str(row)
	# print '------------------------'
	while not valid_move:
		selected_x, selected_y = generateRandxy(board)
		# print selected_x, selected_y
		# Check if this random pair was chosen before
		while [selected_x, selected_y] in random_selections:
			selected_x, selected_y = generateRandxy(board)
		random_selections.append([selected_x, selected_y])
		if board[selected_x][selected_y] == 0:
			valid_move = 1

	return selected_x, selected_y

def play_game():
	n_games = 5000
	learning_rate = 0.01
	discount_factor = 0.75
	q_learning_scores = []
	random_player_scores = []
	
	board1 = init_board()
	board_flatten1 = flatten_board(board1)
	board_size1 = len(board_flatten1)
	all_board_indices1 = range(board_size1)
	
	# Initializes the q table contains all possible states and all possible moves at each state
	# print 'sdfvsd'
	# Or you could load the q table and all states
	q_learning_2_2_data = open("q_learning_2_2_data.pkl","rb")
	q_learning_2_2_data = pickle.load(q_learning_2_2_data)
	q_table = q_learning_2_2_data[0]
	epsilon = 0

	#Count wins
	q_learning_player_win_count = 0
	random_player_win_count = 0
	q_learning_scores_sum = 0
	random_player_scores_sum = 0

	#Initializaing as random values
	# q_table1,all_states = init_q_table()
	# q_table,nStates = init_q_table(board_size1)
	# epsilon = 1
	# print 'sdfvsd'
	state_to_board_mapping = create_state_to_board_mapping()
	epsilonScaled = 0
	# print epsilonScaled
	# print q_table
	# print k
	# raw_input('press enter')
	board = init_board()
	score = init_score()

	# Board will be used later
	board_flatten = flatten_board(board)
	board_size = len(board_flatten)
	
	# Draw the initialize state of the board
	if enable_viz:
		draw_board(board, score)
	sleep(5)
	# Variable to keep track of current player
	current_player = 1 # alternates between 1 and 2
	
	def check_and_set_square(x, y):    
		if score[x][y]:
			return 0
		
		if (board[x * 2][y] and         # above
			board[x * 2 + 1][y] and     # left
			board[x * 2 + 1][y + 1] and # right
			board[x * 2 + 2][y]):       # bottom 
				score[x][y] = current_player
				return 1
		return 0
	
	selected_x, selected_y = 0, 0
	filled = False
	# print check_board_full(board)
	while 1:
		sleep(0.5)
		draw_flag = True

		# sleep(0.1)
		# Select the closest free
		# print 'continue'
		prev_state = flatten_board(board)
		# if prev_state in q_table:

		# else:

		# prev_state_index = all_states.index(prev_state)
		curr_state = flatten_board(board)
		if current_player==1:
			selected_x, selected_y = getRandMove(board)

		else:
			if str(curr_state) in q_table:
				# print 'executing q learning move'
				#Selecting next move using q table
				#Finding the next action by finding the action with minimum weight in the curretn state in the q table
				#Find the current state
				
				# print board
				# print curr_state
				# curr_state_index = all_states.index(curr_state)
				# print curr_state_index

				# Find the possible moving locations in this state
				# This could be made faster by storing possible actions in an array and then accessing it
				# Note: If it is slow, this can be removed and it would still work because we are checking for min
				possible_action_indices = find_possible_actions(curr_state,board_size)
				# print possible_action_indices
				min_weight_index = find_max_index(q_table,curr_state,possible_action_indices,epsilonScaled)
				# print q_table[curr_state_index,:]
				# print min_weight_index
				# print state_to_board_mapping[min_weight_index]

				selected_x = state_to_board_mapping[min_weight_index][0]
				selected_y = state_to_board_mapping[min_weight_index][1]

				# print 'selected row, col'
				# print selected_x
				# print selected_y
				# quit()
			else:
				# raw_input('asdf')
				selected_x, selected_y = getRandMove(board)



		# Board functions

		if enable_viz:
			draw_board(board, score)
			
		x1,y1 = selected_x,selected_y
			
		filled_boxes = 0
		prev_board = board
		board[selected_x][selected_y] = 1
		
		if selected_x % 2 == 0:
			# Now it's time to check for new boxes
			# o   o   
			#        
			# o---o  A new horizontal line can only fill in above
			#        or below
			# o   o
			
			if x1 > 0:
				# check above and upadte score
				filled_boxes += check_and_set_square((x1-1)/2, y1)
			
			if (x1-1)/2+1 < BOARD_SIZE-1:
				# check above and upadte score
				filled_boxes += check_and_set_square((x1-1)/2+1, y1)
		else:
			# Now it's time to check for new boxes
			# o   o   o   A new vertical line can only fill in
			#     |       on the left or right
			# o   o   o
			
			if y1 > 0:
				# check left and upadte score
				filled_boxes += check_and_set_square((x1-1)/2, y1-1)
			
			if y1 < BOARD_SIZE - 1:
				# check right  and upadte score
				filled_boxes += check_and_set_square((x1-1)/2, y1)
		# print 'filled_boxes'
		# print filled_boxes
		prev_player = current_player
		# if check_board_full(board):
		# 	break
		prev_player = current_player
		if filled_boxes==0 and filled!=True:
			# switch players
			current_player = current_player % 2 + 1				

		curr_state = flatten_board(board)
		# curr_state_index = all_states.index(curr_state)
		# prev_action_index = state_to_board_mapping.index([selected_x,selected_y])
		# Updating q table
		# print 'filled'
		# print filled
		# print 'filled_boxes'
		# print filled_boxes
		# print 'draw_flag'
		# print draw_flag
		# print 'reward'
		# print reward
		# raw_input('asdfasd')
		# raw_input("Press Enter to continue...")
		if enable_viz:
			draw_board(board, score)
			stdscr.refresh()
		if check_board_full(board):
			break
		# print '-----------------------------'
		
	sleep(3)
	# random_player_score,q_learning_score = get_final_score(score)
	# # print 'final scores'
	# # print q_learning_score,random_player_score
	# if q_learning_score==random_player_score:
	# 	'draw'
	# elif random_player_score>q_learning_score:
	# 	# print 'random won'
	# 	random_player_win_count += 1
	# elif q_learning_score>random_player_score:
	# 	# print 'q learning won'
	# 	q_learning_player_win_count += 1
	# # raw_input('asdfasd')
	# q_learning_scores.append(q_learning_score)
	# random_player_scores.append(random_player_score)
	# q_learning_scores_sum += q_learning_score
	# random_player_scores_sum += random_player_score
	
	# #Update win counter
	# #Count wins
	# # print current_player
	# # print 'final scores'
	# # print q_learning_score,random_player_score
	# # print 'win count'
	# # print 'q learning player'
	# # print player_2_win_count
	# # print 'random player'
	# # print player_1_win_count
	# # print 'scores sum'
	# # print 'q learning player'
	# # print player_2_scores_sum
	# # print 'random player'
	# # print player_1_scores_sum
	# # raw_input('asdas')
	# # print 'EXIT'
	# # sleep(3)

	# # print q_learning_scores
	# # print random_player_scores
	# q_learning_scores = [float(i) for i in q_learning_scores]
	# random_player_scores = [float(i) for i in random_player_scores]
	# # q_learning_scores_sum = np.sum(q_learning_scores)
	# # random_player_scores_sum = np.sum(random_player_scores)
	# # print 'win count'
	# # print 'q learning player'
	# # print q_learning_player_win_count
	# # print 'random player'
	# # print random_player_win_count
	# # print 'scores sum'
	# # print 'q learning player'
	# # print q_learning_scores_sum
	# # print 'random player'
	# # print random_player_scores_sum

	# if len(q_learning_scores)>100:
	# 	short_q_learning_scores = []
	# 	short_random_player_scores = []
	# 	nGamesScaled = n_games/100
	# 	scale = n_games/nGamesScaled
	# 	for k in range(scale):
	# 		# print k*nGamesScaled,(k+1)*nGamesScaled
	# 		# print q_learning_scores[k*nGamesScaled:(k+1)*nGamesScaled]
	# 		q_learning_avg_score = np.sum(q_learning_scores[k*nGamesScaled:(k+1)*nGamesScaled])/nGamesScaled
	# 		random_player_avg_score = np.sum(random_player_scores[k*nGamesScaled:(k+1)*nGamesScaled])/nGamesScaled
	# 		# print q_learning_avg_score
	# 		# print random_player_avg_score
	# 		# raw_input('asda')
	# 		short_q_learning_scores.append(q_learning_avg_score)
	# 		short_random_player_scores.append(random_player_avg_score)
	# 	plt.title('100000 games trained q learning player')
	# 	plt.plot(short_q_learning_scores, label='Q learning player')
	# 	plt.plot(short_random_player_scores,'--', label='Random player')
	# 	plt.ylabel('final scores after each game')
	# 	plt.xlabel('game number')
	# 	plt.legend()
	# 	plt.show()
	# else:
	# 	plt.title('100000 games trained q learning player')
	# 	plt.plot(q_learning_scores, label='Q learning player')
	# 	plt.plot(random_player_scores,'--', label='Random player')
	# 	plt.ylabel('final scores after each game')
	# 	plt.xlabel('game number')
	# 	plt.legend()
	# 	plt.show()

def draw_board(board, scores):
	# +-------> y-axis
	# |   1 2
	# | A . .
	# | B . .
	# v
	# x-axis
	#
	# draw the headers
	
	# now draw the board

	# draws the dots and lines corresponding to the vertical lines
	for i, row in enumerate(board):
		for j, column in enumerate(row):
			# print "here %d, %d" % (i, j)
			if i % 2 == 0:
				draw_dot(j, i)
			if column:
				draw_line(j, i)
	
	stdscr.refresh()
	
	# We need to draw the letters in the boxes too and add the score
	player1, player2 = 0, 0
	for i, row in enumerate(scores):
		for j, score in enumerate(row):
			if score == 1: player1 += 1
			if score == 2: player2 += 1
			if score:
				draw_filling(j, i, score)
	
	# Draw the score board
	screen_y = OFFSET_Y + BOARD_SIZE * 2
	screen_x = OFFSET_X
	stdscr.addstr(screen_y, screen_x, "Random Player: %d" % player1, 
					curses.color_pair(1))
	stdscr.addstr(screen_y + 1, screen_x, "Q Learning player: %d" % player2, 
					curses.color_pair(2))
	
	stdscr.refresh()


if __name__ == '__main__':
	try:
		play_game()
	except KeyboardInterrupt:
		pass
	finally:
		if enable_viz:
			curses.endwin()
