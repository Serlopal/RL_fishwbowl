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
from cv2 import resize, INTER_CUBIC, imwrite, INTER_NEAREST

import copy
import numpy as np
import tensorflow as tf
import pyqtgraph as pg
import cv2


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
		self.frame_size = 32

		# ----------------------------
		self.state_size = state_size
		self.action_size = 8
		self.memory = deque(maxlen=200000)
		self.gamma = 0.99  # discount rate
		self.epsilon = 1.0  # 1.0  # exploration rate
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.99996
		self.curr_state = None
		self.curr_action = 8
		self.speed = 0.05

		self.update_target_freq = 10000

		self.wlen = 4
		self.frame_memory = deque(maxlen=self.wlen)
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
		n = 4 if reward > 0 else 1
		for _ in range(n):
			self.save_to_memory(state, action, reward, next_state, terminal_flag)

	def build_state(self):
		# now build cube
		stacked_memory = np.stack(self.frame_memory, axis=2)
		# stacked_memory = np.stack(self.frame_memory, axis=0)
		return np.expand_dims(stacked_memory, axis=0)

	def build_model(self):
		if os.path.exists("model.h5"):
			print("found previous model	")
			model = load_model(self.model_name, custom_objects={'huber_loss': tf.losses.huber_loss})
			return model
		else:
			# Neural Net for Deep-Q learning Model
			model = Sequential()
			model.add(Conv2D(16, kernel_size=4, strides=2, activation='relu'))
			model.add(Conv2D(32, kernel_size=2, strides=1, activation='relu'))
			model.add(Flatten())
			model.add(Dense(128, activation='relu'))
			model.add(Dense(self.action_size, activation='linear'))
			model.compile(loss=tf.losses.huber_loss, optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
			return model

	def save_model(self):
		self.model.save(self.model_name)

	def save_to_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def choose_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size, t):

		def get_target_f(sample):
			state, action, reward, next_state, done = sample
			if done:
				target = reward
			else:
				target = reward + self.gamma * np.clip(np.amax(self.target_model.predict(next_state)[0]), -1, 1)

			target_f = self.model.predict(state)
			target_f[0][action] = target
			return target_f

		minibatch = random.sample(self.memory, batch_size)

		#  minibatch = self.memory
		target_fs = list(map(get_target_f, minibatch))

		#  zip together all states, actions, rewards, next_states, dones
		states, actions, rewards, next_states, dones = map(list, list(zip(*minibatch)))
		self.model.fit(np.concatenate(states, axis=0), np.concatenate(target_fs, axis=0), epochs=1, verbose=False, batch_size=32)
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

	def process_frame(self, frame):
		frame = frame[int(0.15*frame.shape[0]):-int(0.15*frame.shape[0]), int(0.3*frame.shape[1]):-int(0.3*frame.shape[1])]
		frame = resize(frame, (self.frame_size, self.frame_size), interpolation=INTER_NEAREST )
		# cv2.imwrite("asd.jpg", frame)
		return frame/255

	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())


class Fishbowl(QWidget):
	animation_emitter = pyqtSignal(object)

	def __init__(self, info_signal, viewer_signal):
		super().__init__()

		# connect signal from emitter to trigger the animation
		self.animation_emitter.connect(lambda x: self.life_loop(x))

		self.fishbowl_color = QColor(128, 128, 128)
		self.fishbowl_radius = 1.0
		self.fishbowl_pixel_radius = 100
		self.wheel_size = self.fishbowl_pixel_radius * 1.15
		self.wheel_width = self.fishbowl_pixel_radius * 0.15

		self.npc_pixel_radius = self.fishbowl_pixel_radius * 0.05
		self.player_pixel_radius = self.fishbowl_pixel_radius * 0.075

		self.fishbowl_border_size = self.fishbowl_pixel_radius * 0.004
		self.fishbowl_thin_border_size = self.fishbowl_pixel_radius * 0.002

		self.nenemies = 6
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
		self.observe_iterations = 2000
		self.max_episode_len = 400
		self.episode_t = 0
		self.global_t = 0

		self.n_games = 1
		self.info_signal = info_signal
		self.viewer_signal = viewer_signal

		self.screen = QGuiApplication.primaryScreen()

	def scale_point(self, point):
		new_max = self.fishbowl_pixel_radius
		return ((p / self.fishbowl_radius) * new_max for p in point)

	def life_loop(self, command):
		"""
		:param command: unused
		this method imlements the main loop of the fishbowl
		in which we move around players, check who is dead and so on
		"""

		if command == "act":
			# get first frames into memory
			frame = self.player.process_frame(self.get_frame())
			self.player.frame_memory.append(frame)

			# build current state
			state = self.player.build_state()

			# store state before moving
			self.player.curr_state = state

			if len(self.player.memory) < self.observe_iterations:
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

			# if episode has taken too much time, end it
			if self.episode_t > self.max_episode_len:
				self.restart_game()
				return

			# observe state after movement
			frame = self.player.process_frame(self.get_frame())

			# update memory stack with new frame
			self.player.frame_memory.append(frame)
			# build next state
			next_state = self.player.build_state()

			# check if the state is terminal
			terminal_flag = self.player.check_killed(enemies=self.enemies)

			# build reward depending on terminal flag
			if terminal_flag:
				reward = -1
			else:
				reward = +1

			# add reward to episode sum for performance tracking
			self.episode_reward += reward

			# store movement experience to memory
			self.player.remember(self.player.curr_state, self.player.curr_action, reward, next_state, terminal_flag)

			# increase global and episodic counter
			self.global_t += 1
			self.episode_t += 1

			# update target network if it is time to
			if self.global_t % self.player.update_target_freq == 0:
				print("updating target network")
				self.player.update_target_model()

			# check game is over. If so, train the model if our memory is populated
			if terminal_flag:
				if len(self.player.memory) > self.observe_iterations:
					self.player.replay(batch_size=32, t=self.global_t)
					if self.n_games % 300 == 0:
						self.player.save_model()
				self.restart_game()
				return

	def get_frame(self):
		frame = self.screen.grabWindow(self.winId()).toImage().convertToFormat(QImage.Format_Grayscale8)
		h, w = frame.height(), frame.width()
		s = frame.bits().asstring(w * h * 1)
		arr = np.fromstring(s, dtype=np.uint8).reshape((h, w, 1))

		return arr

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
			for _ in range(3):
				self.animation_emitter.emit("repeat_action")
			time.sleep(0.000001)
			self.animation_emitter.emit("learn")

	def restart_game(self):
		self.n_games += 1
		self.info_signal.emit("Game {0} - Exploration Rate {1} - {2}".format(
			self.n_games, np.round(self.player.epsilon, 5), "Training" if len(self.player.memory) > self.observe_iterations else "Memory size {}".format(len(self.player.memory))))
		for enemy in self.enemies:
			enemy.revive()
		self.player.revive()
		# save reward for espisode
		self.reward_memory.append(self.episode_reward)
		# emit and reset episode reward
		self.episode_reward = 0.0
		self.episode_t = 0


class gameUI:

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

	ui = gameUI()
	ui.start_ui()
