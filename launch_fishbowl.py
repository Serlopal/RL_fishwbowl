from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QProgressBar, QComboBox, QDesktopWidget, \
	QGridLayout, QSlider, QGroupBox, QVBoxLayout, QHBoxLayout, QStyle, QScrollBar, QMainWindow, QAction, QDialog
from PyQt5.QtCore import QDateTime, Qt, QTimer, QPoint, pyqtSignal, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPalette, QScreen, QImage, QGuiApplication
import time
import numpy as np
import threading
from collections import deque
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, Conv3D, GRU
from keras.optimizers import *
from keras.models import load_model
import random
import os
import tensorflow as tf
import pyqtgraph as pg


class QSignalViewer(pg.PlotWidget):
	emitter = pyqtSignal(object)

	def __init__(self, num_signals):
		super().__init__()
		# save number of signals
		self.nplots = num_signals
		# set number of samples to be displayed per signal at a time
		self.nsamples = 20000
		# connect the signal to be emitted by the feeder to the slot of the plotWidget that will update the signals
		self.emitter.connect(lambda values: self.update(values))
		# buffer to store the data from all signals
		self.buff = np.zeros((self.nplots, 0))
		# create curves for the signals
		self.curves = []
		for i in range(self.nplots):
			c = pg.PlotCurveItem(pen=(i, self.nplots * 1.3))
			self.addItem(c)
			self.curves.append(c)

	def update(self, data):
		# update buffer
		if len(self.buff) > self.nsamples:
			self.buff = np.concatenate([self.buff[:, 1:], np.reshape(data, (-1, 1))], axis=1)
		else:
			self.buff = np.concatenate([self.buff, np.reshape(data, (-1, 1))], axis=1)
		# update plots
		for i in range(self.nplots):
			self.curves[i].setData(self.buff[i])
		self.repaint()

	def update_signals(self, values):
		self.emitter.emit(values)


class DynamicLabel(QLabel):
	signal = pyqtSignal(object)

	def __init__(self, base_text):
		super().__init__()
		self.font = QFont("Times", 12, QFont.Bold)
		self.setFont(self.font)

		self.base_text = base_text
		self.setText(self.base_text + "0")
		self.signal.connect(lambda x: self.setText(x))


class NPC():
	allowed_dirs = [np.array(x) for x in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]]

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius):
		self.original_coords = np.array([0, 0])
		self.r = pixel_radius / fishbowl_pixel_radius
		self.fishbowl_radius = fishbowl_radius
		self.original_color = Qt.black
		self.color = Qt.black
		self.coords = self.original_coords
		self.v = np.array([0, 0])
		self.speed = 0.05
		self.pvdi = np.array([0, 0])
		self.first = True
		self.dead = False

	def move(self):
		self.coords = self.spherical_clip(self.coords + self.v)
		if not self.inside_sphere() or self.first:
			self.first = False
			angle = np.deg2rad(np.random.randint(low=0, high=361, size=1))
			self.v = self.speed * np.reshape([np.cos(angle), np.sin(angle)], (1, -1))

	def move_fixed_angle(self):
		self.coords = self.spherical_clip(self.coords + self.v)
		if not self.inside_sphere() or self.first:
			if self.first:
				angle = np.deg2rad(np.random.randint(low=0, high=361, size=1))
				self.v = self.speed * np.reshape([np.cos(angle), np.sin(angle)], (1, -1))
				self.first = False
			else:
				self.v *= np.array([-1, -1])

	@staticmethod
	def dist(a, b):
		return np.sqrt((a[0, 0] - b[0, 0]) ** 2 + (a[0, 1] - b[0, 1]) ** 2)

	def revive(self):
		self.color = self.original_color
		self.coords = self.spherical_clip(np.reshape((np.array([np.random.rand(), np.random.rand()]) * 2) - 1, (1, -1)))
		self.v = np.array([0, 0])
		self.first = True
		self.dead = False

	def spherical_clip(self, point):
		dist = np.sqrt(np.sum(point ** 2))
		if dist > self.fishbowl_radius - self.r:
			point = point * ((self.fishbowl_radius - self.r) / dist)
		return  point

	def inside_sphere(self):
		dist = np.sqrt(np.sum(self.coords ** 2))
		if dist > (self.fishbowl_radius - self.r):
			return False
		else:
			return True


class Enemy(NPC):

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, color):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.original_color = color
		self.color = self.original_color
		self.original_coords = self.spherical_clip(np.reshape([np.random.rand(), np.random.rand()], (1, -1)) * 2 - 1)
		self.coords = self.original_coords


class Reward(NPC):
	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, color):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.original_color = color
		self.color = self.original_color
		self.original_coords = self.spherical_clip(np.reshape([np.random.rand(), np.random.rand()], (1, -1)) * 2 - 1)
		self.coords = self.original_coords


class Player(NPC):
	"""
	https://keon.io/deep-q-learning/
	"""

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, state_size):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		# self.allowed_dirs.append(np.array([0, 0]))

		self.original_coords = self.spherical_clip(np.reshape((np.array([np.random.rand(), np.random.rand()]) * 2) - 1, (1, -1)))
		self.original_color = Qt.black
		self.color = self.original_color
		self.coords = self.original_coords
		self.frame_size = 84
		self.frame_skip = 4

		# ----------------------------
		self.state_size = state_size
		self.action_size = 8
		self.memory = deque(maxlen=1000000)
		self.gamma = 0.99  # discount rate
		self.epsilon = 1.0  # 1.0  # exploration rate
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.99998
		self.curr_state = None
		self.curr_action = 8
		self.speed = 0.05

		self.update_target_freq = 10000
		self.save_model_step = 4000
		self.observe_iterations = 20000

		self.qvalue_example = 0.0
		self.wlen = 4
		self.state_memory = deque(maxlen=self.wlen)
		self.model_name = "model.h5"
		self.model = self.build_model()
		self.target_model = self.build_model()

	def act(self, state=None, repeat_action=False, given_action=None):  # TODO apparently PEP8 does not let you change the args in an inhereted method...
		if state is not None:
			# choose an action
			action = self.choose_action(state)
			# update player coords
			self.coords = self.spherical_clip(self.coords + self.allowed_dirs[action] * self.speed)

			return action
		elif repeat_action:
			# update player coords
			self.coords = self.spherical_clip(self.coords + self.allowed_dirs[self.curr_action] * self.speed)
		elif given_action is not None:
			# update player coords
			self.coords = self.spherical_clip(self.coords + self.allowed_dirs[given_action] * self.speed)
		else:
			raise Exception("player needs state or action to act")

	def remember(self, state, action, reward, next_state, terminal_flag):
		# increase the probability of terminal states being used for training
		n = 4 if reward < 0 else 1
		for _ in range(n):
			self.save_to_memory(state, action, reward, next_state, terminal_flag)

	def build_model(self):
		if os.path.exists("model.h5"):
			print("found previous model	")
			model = load_model(self.model_name, custom_objects={'huber_loss': tf.losses.huber_loss})
			return model
		else:
			# Neural Net for Deep-Q learning Model
			model = Sequential()
			model.add(GRU(128, input_shape=(self.wlen, self.state_size), return_sequences=False))
			model.add(Dense(64, activation='relu'))
			model.add(Dense(32, activation='relu'))
			model.add(Dense(self.action_size, activation='linear'))
			# opt = RMSprop(lr=0.025, rho=0.95, epsilon=0.01)
			opt = Adam()
			model.compile(loss=tf.losses.huber_loss, optimizer=opt)
			global graph
			graph = tf.get_default_graph()
			return model

	def save_model(self):
		self.model.save(self.model_name)

	def save_to_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def choose_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		with graph.as_default():
			act_values = self.model.predict(state)
			self.qvalue_example = act_values
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size, t):
		with graph.as_default():
			def get_target_f(sample):
				state_pred, action, reward, next_state_pred, done = sample
				if done:
					target = reward
				else:
					target = reward + self.gamma * np.clip(np.amax(next_state_pred[0]), -1, 1)

				target_f = state_pred
				target_f[action] = target
				return target_f

			def states2targets(minibatch):
				states, actions, rewards, next_states, dones = zip(*minibatch)
				state_preds = self.model.predict(np.concatenate(states, axis=0))
				next_state_preds = self.target_model.predict(np.concatenate(next_states, axis=0))
				return list(zip(state_preds, actions, rewards, next_state_preds, dones))

			minibatch = random.sample(self.memory, batch_size)
			# to speed up performance, do predictions for state and next state first
			minibatch_with_preds = states2targets(minibatch)
			target_fs = list(map(get_target_f, minibatch_with_preds))

			#  zip together all states, actions, rewards, next_states, dones
			states, actions, rewards, next_states, dones = map(list, list(zip(*minibatch)))
			self.model.fit(np.concatenate(states), np.stack(target_fs, axis=0), epochs=1, verbose=False, batch_size=batch_size)
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

	def revive(self):
		super().revive()

	def check_killed(self, enemies):
		p = self.coords
		for enemy in enemies:
			e = enemy.coords
			dist = self.dist(e, p)
			if dist < (self.r + enemy.r):  # two balls touching, 2 radius
				return True
		return False

	@staticmethod
	def process_state(state):
		return state.flatten()

	def build_state(self):
		# now build cube
		stacked_memory = np.stack(self.state_memory, axis=0)
		return np.expand_dims(stacked_memory, axis=0)

	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())


class Fishbowl(QWidget):
	animation_emitter = pyqtSignal(object)

	def __init__(self, info_signal, viewer_signal):
		super().__init__()

		# connect signal from emitter to trigger the animation
		self.animation_emitter.connect(lambda x: self.life_loop_train(x))

		self.fishbowl_color = QColor(128, 128, 128)
		self.fishbowl_radius = 1.0
		self.fishbowl_pixel_radius = 100
		self.wheel_size = self.fishbowl_pixel_radius * 1.15
		self.wheel_width = self.fishbowl_pixel_radius * 0.15

		self.npc_pixel_radius = self.fishbowl_pixel_radius * 0.05
		self.player_pixel_radius = self.fishbowl_pixel_radius * 0.075

		self.fishbowl_border_size = self.fishbowl_pixel_radius * 0.004
		self.fishbowl_thin_border_size = self.fishbowl_pixel_radius * 0.002

		self.nenemies = 4
		colors = [Qt.red, Qt.green, Qt.blue, Qt.yellow, Qt.cyan, Qt.magenta]
		self.enemies = [Enemy(
			self.npc_pixel_radius, self.fishbowl_pixel_radius, self.fishbowl_radius, Qt.white) for i in range(self.nenemies)]

		self.curr_reward_flag = False

		self.player = Player(
			self.npc_pixel_radius,
			self.fishbowl_pixel_radius,
			self.fishbowl_radius,
			(self.nenemies + 1) * 2)

		self.episode_reward = 0
		self.reward_memory = deque(maxlen=100)
		self.episode_t = 0
		self.global_t = 0

		self.n_games = 1
		self.info_signal = info_signal
		self.viewer_signal = viewer_signal
		self.episode_time = time.time()

		self.screen = QGuiApplication.primaryScreen()

	def scale_point(self, point):
		new_max = self.fishbowl_pixel_radius
		return ((p / self.fishbowl_radius) * new_max for p in point)

	def life_loop_train(self, command):
		"""
		:param command: unused
		this method imlements the main loop of the fishbowl
		in which we move around players, check who is dead and so on
		"""

		if command == "act":
			frame = self.player.process_state(self.get_state())
			self.player.state_memory.append(frame)

			# build current state
			state = self.player.build_state()

			# store state before moving
			self.player.curr_state = state

			if len(self.player.memory) < self.player.observe_iterations:
				action = np.random.randint(low=0, high=self.player.action_size)
				self.player.act(given_action=action)
			else:
				# issue player action
				action = self.player.act(state)

			# save chosen action
			self.player.curr_action = action

			# issue enemies reaction
			for enemy in self.enemies:
				enemy.move()

			# render new frame state
			self.repaint()

		elif command == "repeat_action":
			# check game has finished
			terminal_flag = self.player.check_killed(enemies=self.enemies)
			if terminal_flag:
				return

			# issue same action as before
			self.player.act(repeat_action=True)

			# issue enemies reaction
			for enemy in self.enemies:
				enemy.move()

			# render new frame state
			self.repaint()

		elif command == "learn":
			# observe state after movement
			frame = self.player.process_state(self.get_state())

			# update memory stack with new frame
			self.player.state_memory.append(frame)
			# build next state
			next_state = self.player.build_state()

			# check if the state is terminal
			terminal_flag = self.player.check_killed(enemies=self.enemies)

			# build reward depending on terminal flag
			if terminal_flag:
				reward = -1
			else:
				reward = +0.01

			# add reward to episode sum for performance tracking
			self.episode_reward += reward

			# store movement experience to memory if the player has already seen enough frames in a row to build a state
			if len(self.player.state_memory) >= self.player.wlen:
				self.player.remember(self.player.curr_state, self.player.curr_action, reward, next_state, terminal_flag)

			# increase global and episodic counter
			self.global_t += 1
			self.episode_t += 1

			# update target network if it is time to
			if self.global_t % self.player.update_target_freq == 0 and len(self.player.memory) > self.player.observe_iterations:
				print("updating target network")
				self.player.update_target_model()

			# check game is over. If so, train the model if our memory is populated
			if terminal_flag:
				if len(self.player.memory) > self.player.observe_iterations:
					self.player.replay(batch_size=32, t=self.global_t)
					if self.n_games % self.player.save_model_step == 0:
						self.player.save_model()
				self.restart_game()
				return

	def life_loop_test(self, command):
		"""
		:param command: unused
		this method imlements the main loop of the fishbowl
		in which we move around players, check who is dead and so on
		"""

		if command == "act":
			frame = self.player.process_state(self.get_state())
			self.player.state_memory.append(frame)

			# build current state
			state = self.player.build_state()

			# issue player action
			action = self.player.act(state)

			# issue enemies reaction
			for enemy in self.enemies:
				enemy.move()

			# render new frame state
			self.repaint()

		elif command == "repeat_action":
			# check game has finished
			terminal_flag = self.player.check_killed(enemies=self.enemies)
			if terminal_flag:
				return

			# issue same action as before
			self.player.act(repeat_action=True)

			# issue enemies reaction
			for enemy in self.enemies:
				enemy.move()

			# render new frame state
			self.repaint()

		elif command == "check_terminal":
			# observe state after movement
			frame = self.player.process_state(self.get_state())

			# update memory stack with new frame
			self.player.state_memory.append(frame)

			# check if the state is terminal
			terminal_flag = self.player.check_killed(enemies=self.enemies)

			# check game is over. If so, train the model if our memory is populated
			if terminal_flag:
				self.restart_game()

	def get_state(self):
		return np.vstack([x.coords for x in self.enemies] + [self.player.coords])

	def paintEvent(self, e):
		qp = QPainter()
		qp.begin(self)
		self.drawWidget(qp)
		qp.end()

	def drawWidget(self, qp):
		c = self.rect().center()
		c_coords = c.x(), c.y()
		background_color = self.palette().color(QPalette.Background)

		# paint fishbowl
		qp.setPen(QPen(self.fishbowl_color, self.fishbowl_border_size, Qt.SolidLine))
		# draw fishbowl
		qp.setBrush(QBrush(self.fishbowl_color, Qt.SolidPattern))
		qp.drawEllipse(c, *([self.fishbowl_pixel_radius] * 2))

		# draw axis lines
		# qp.setPen(QPen(self.fishbowl_color, self.fishbowl_thin_border_size, Qt.DashDotDotLine))
		# for angle in range(0, 420, 45):
		# 	line = QLineF(); line.setP1(c); line.setAngle(angle); line.setLength(self.fishbowl_pixel_radius)
		# 	qp.drawLine(line)
		# # draw wheel separators
		# line = QLineF(); line.setP1(c + QPoint(self.wheel_size, 0)); line.setAngle(0); line.setLength(self.wheel_width)
		# qp.drawLine(line)
		# line = QLineF(); line.setP1(c + QPoint(0, -self.wheel_size)); line.setAngle(90); line.setLength(self.wheel_width)
		# qp.drawLine(line)
		# line = QLineF(); line.setP1(c + QPoint(-self.wheel_size, 0)); line.setAngle(180); line.setLength(self.wheel_width)
		# qp.drawLine(line)
		# line = QLineF(); line.setP1(c + QPoint(0, self.wheel_size)); line.setAngle(270); line.setLength(self.wheel_width)
		# qp.drawLine(line)

		qp.setPen(QPen(self.fishbowl_color, self.fishbowl_border_size, Qt.SolidLine))

		# draw enemies
		for i, enemy in enumerate(self.enemies):
			qp.setBrush(QBrush(enemy.color, Qt.SolidPattern))
			qp.drawEllipse(c + QPoint(*self.scale_point(np.squeeze(enemy.coords))), *([self.npc_pixel_radius] * 2))

		# draw player
		qp.setBrush(QBrush(self.player.color, Qt.SolidPattern))
		qp.drawEllipse(c + QPoint(*self.scale_point(np.squeeze(self.player.coords))), *([self.player_pixel_radius] * 2))

	def animate_balls(self):
		self.update_thread = threading.Thread(target=self._animate_balls)
		self.update_thread.daemon = True
		self.update_thread.start()

	def _animate_balls(self):
		time.sleep(2)
		while True:
			self.animation_emitter.emit("act")
			time.sleep(0.03)
			for _ in range(self.player.frame_skip - 1):
				self.animation_emitter.emit("repeat_action")
				time.sleep(0.03)
			self.animation_emitter.emit("check_terminal")

	def animate_balls_noqt(self):
		time.sleep(2)
		for _ in range(self.player.wlen):
			self.life_loop_train("act")
			for _ in range(self.player.frame_skip - 1):
				self.life_loop_train("repeat_action")

			while True:
				self.life_loop_train("act")
				for _ in range(self.player.frame_skip - 1):
					self.life_loop_train("repeat_action")
				self.life_loop_train("learn")

	def restart_game(self):
		self.n_games += 1
		message = "Game {0} - Exploration Rate {1} - {2} - episode reward {3: < 6} - duration {4: < 6} - last qvalues {5}".format(
			self.n_games,
			np.round(self.player.epsilon, 5),
			"Training" if len(self.player.memory) > self.player.observe_iterations else "Memory size {}".format(len(self.player.memory)),
			np.round(self.episode_reward, 4),
			np.round(time.time() - self.episode_time, 4),
			np.round(self.player.qvalue_example, 2)
		)
		print(message)
		# self.info_signal.emit(message)
		for enemy in self.enemies:
			enemy.revive()
		self.player.revive()
		# save reward for espisode
		self.reward_memory.append(self.episode_reward)
		# emit and reset episode reward
		self.episode_reward = 0.0
		self.episode_t = 0
		# set new episode start time
		self.episode_time = time.time()


class GameUI:

	def __init__(self, UI_name="fishbowl"):
		self.app = QApplication([UI_name])
		self.app.setObjectName(UI_name)
		self.UI_name = UI_name
		# set app style
		self.app.setStyle("Fusion")
		# create main window
		self.window = QMainWindow()
		self.window.setWindowTitle(UI_name)
		self.window.setObjectName(UI_name)
		self.main_group = QGroupBox()
		self.window.setCentralWidget(self.main_group)
		# set window geometry
		ag = QDesktopWidget().availableGeometry()
		self.window.move(int(ag.width()*0.15), int(ag.height()*0.05))
		self.window.setMinimumWidth(int(ag.width()*0.3))
		self.window.setMinimumHeight(int(ag.height()*0.4))

		self.layout = QGridLayout()
		self.n_games_label = DynamicLabel("Game ")
		self.signal_viewer = QSignalViewer(1)
		self.fishbowl = Fishbowl(self.n_games_label.signal, self.signal_viewer.emitter)

		self.layout.addWidget(self.n_games_label, 0, 0, 1, 10)
		self.layout.addWidget(self.fishbowl, 1, 0, 10, 10)
		# self.layout.addWidget(self.signal_viewer, 1, 10, 10, 10)

		self.main_group.setLayout(self.layout)

		# set layout inside window
		self.window.setLayout(self.layout)
		self.window.show()

	def start_ui(self):
		"""
		starts the ball animation thread and launches the QT app
		"""
		self.start_animation()
		self.app.exec()

	def start_animation(self):
		"""
		waits 1 second so that the QT app is running and then launches the ball animation thread
		"""
		self.fishbowl.animate_balls()


if __name__ == "__main__":
	ui = GameUI()
	ui.start_ui()
