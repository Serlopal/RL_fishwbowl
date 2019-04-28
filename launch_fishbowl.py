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
from keras.optimizers import Adam
from keras.models import load_model
import random
import os
from cv2 import resize, INTER_CUBIC, imwrite

import copy
import numpy as np
import tensorflow as tf

def huber_loss(y_true, y_pred):
	return tf.losses.huber_loss(y_true, y_pred)


def qt_image_to_array(img, share_memory=False):
    """ Creates a numpy array from a QImage.

        If share_memory is True, the numpy array and the QImage is shared.
        Be careful: make sure the numpy array is destroyed before the image,
        otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QImage), "img must be a QtGui.QImage object"
    assert img.format() == QImage.Format.Format_RGB32, \
        "img format must be QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image = img_size.width() * img_size.height() * img.depth()
    assert n_bits_buffer == n_bits_image, \
        "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape=(img_size.height(), img_size.width(), img.depth()//8),
                     buffer=buffer,
                     dtype=np.uint8)

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)


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
		self.speed = 0.1
		self.pvdi = np.array([0, 0])
		self.first = True
		self.dead = False

	def move(self):
		self.coords = self.spherical_clip(self.coords + self.v)
		if not self.inside_sphere() or self.first:
			self.first = False
			self.v = np.reshape((np.array([np.random.rand(), np.random.rand()]) * 2) - 1, (1, -1)) * self.speed

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
		self.original_coords = self.spherical_clip(np.reshape((np.array([np.random.rand(), np.random.rand()]) * 2) - 1, (1, -1)))
		self.coords = self.original_coords


class Player(NPC):
	"""
	https://keon.io/deep-q-learning/
	"""

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, state_size):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.allowed_dirs.append(np.array([0, 0]))

		self.original_coords = self.spherical_clip(np.reshape((np.array([np.random.rand(), np.random.rand()]) * 2) - 1, (1, -1)))
		self.original_color = Qt.black
		self.color = self.original_color
		self.coords = self.original_coords
		self.frame_size = 128

		# ----------------------------
		self.state_size = state_size
		self.action_size = 9
		self.memory = deque(maxlen=2000)
		self.coord_history = deque(maxlen=20)
		self.gamma = 0.95  # discount rate
		self.epsilon = 0.0  # 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.curr_state = None
		self.curr_action = None
		self.speed = 0.1
		# self.learning_rate = 0.001
		self.wlen = 4
		self.frame_memory = deque(maxlen=self.wlen)
		self.model_name = "model.h5"
		self.model = self.build_model()

	def act(self, state):  # TODO apparently PEP8 does not let you change the args in an inhereted method...
		"""
		:param frame: all enemy instances to be considered as inputs to determine where the layer shall move
		"""
		# choose an action
		action = self.choose_action(state)

		# update player coords
		self.coords = self.spherical_clip(self.coords + self.allowed_dirs[action] * self.speed)

		return action

	def remember(self, state, action, reward, next_state, terminal_flag):
		self.save_to_memory(state, action, reward, next_state, terminal_flag)

	def build_state(self):
		# now build cube
		stacked_memory = np.stack(self.frame_memory, axis=2)
		# stacked_memory = np.stack(self.frame_memory, axis=0)
		return np.expand_dims(stacked_memory, axis=0)

	def build_model(self):
		if os.path.exists("model.h5"):
			print("found previous model	")
			model = load_model(self.model_name, custom_objects={'huber_loss': huber_loss})
			return model
		else:
			# Neural Net for Deep-Q learning Model
			model = Sequential()
			model.add(Conv2D(64, kernel_size=5, activation='relu'))
			model.add(Conv2D(32, kernel_size=3, activation='relu'))
			model.add(Flatten())
			# model.add(GRU(64, input_shape=(self.wlen, self.state_size), return_sequences=False))
			model.add(Dense(32, activation='relu'))
			# model.add(Dropout(0.1))
			model.add(Dense(self.action_size, activation='linear'))
			model.compile(loss=tf.losses.huber_loss, optimizer=Adam())  #loss=tf.losses.huber_loss, optimizer=Adam())
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

	def replay(self, batch_size):

		def get_target_f(sample):
			state, action, reward, next_state, done = sample
			if done:
				target = reward
			else:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			return target_f

		minibatch = random.sample(self.memory, batch_size)
		#  minibatch = self.memory
		target_fs = list(map(get_target_f, minibatch))

		#  zip together all states, actions, rewards, next_states, dones
		states, actions, rewards, next_states, dones = map(list, list(zip(*minibatch)))

		terminal_index = rewards.index(0) if 0 in rewards else -1
		# if terminal_index != -1:
		# 	terminal_states = states[terminal_index]
		# 	terminal_next_states = next_states[terminal_index]
		# 	imwrite("before1.jpg", terminal_states[0, :, :, 0])
		# 	imwrite("before2.jpg", terminal_states[0, :, :, 1])
		#
		# 	imwrite("after1.jpg", terminal_next_states[0, :, :, 0])
		# 	imwrite("after2.jpg", terminal_next_states[0, :, :, 1])

		self.model.fit(np.concatenate(states, axis=0), np.concatenate(target_fs, axis=0), epochs=1, verbose=False, batch_size=32)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
			print(self.epsilon)

	def revive(self):
		super().revive()
		if len(self.memory) > 32:
			self.memory.clear()
		self.frame_memory.clear()

	def check_killed(self, enemies):
		p = self.coords
		for enemy in enemies:
			e = enemy.coords
			dist = self.dist(e, p)
			if dist < 2*self.r:  # two balls touching, 2 radius
				self.dead = True
				self.color = Qt.gray
				return True
		return False

	def proccess_frame(self, frame):
		frame = frame[int(0.15*frame.shape[0]):-int(0.15*frame.shape[0]), int(0.25*frame.shape[1]):-int(0.25*frame.shape[1])]
		frame = resize(frame, (self.frame_size, self.frame_size), interpolation=INTER_CUBIC)
		return frame


class Fishbowl(QWidget):
	animation_emitter = pyqtSignal(object)

	def __init__(self, info_signal):
		super().__init__()

		# connect signal from emitter to trigger the animation
		self.animation_emitter.connect(lambda x: self.life_loop(x))

		self.fishbowl_color = Qt.gray
		self.fishbowl_radius = 1.0
		self.fishbowl_pixel_radius = 100
		self.wheel_size = self.fishbowl_pixel_radius * 1.15
		self.wheel_width = self.fishbowl_pixel_radius * 0.15
		self.npc_pixel_radius = self.fishbowl_pixel_radius * 0.05
		self.fishbowl_border_size = self.fishbowl_pixel_radius * 0.004
		self.fishbowl_thin_border_size = self.fishbowl_pixel_radius * 0.002

		self.nenemies = 4
		colors = [Qt.red, Qt.green, Qt.blue, Qt.yellow, Qt.cyan, Qt.magenta]
		self.enemies = [Enemy(
			self.npc_pixel_radius, self.fishbowl_pixel_radius, self.fishbowl_radius, colors[i]) for i in range(self.nenemies)]

		self.player = Player(
			self.npc_pixel_radius,
			self.fishbowl_pixel_radius,
			self.fishbowl_radius,
			(self.nenemies + 1) * 2)

		self.n_games = 1
		self.info_signal = info_signal

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
			while len(self.player.frame_memory) < self.player.wlen:
				frame = self.player.proccess_frame(self.get_frame())
				self.player.frame_memory.append(frame)

			# build current state
			state = self.player.build_state()

			# store state
			self.player.curr_state = state

			# issue player action
			action = self.player.act(state)

			# save chosen action
			self.player.curr_action = action

			# issue enemies reaction
			for enemy in self.enemies:
				enemy.move()

			# render new frame state
			self.repaint()
		elif command == "learn":
			# observe next state
			frame = self.player.proccess_frame(self.get_frame())

			# update memory stack with new frame
			self.player.frame_memory.append(frame)

			# build current state
			next_state = self.player.build_state()

			# store action knowledge and check if the state is terminal
			terminal_flag = self.player.check_killed(enemies=self.enemies)
			reward = 0 if terminal_flag else 1

			self.player.remember(self.player.curr_state, self.player.curr_action, reward, next_state, terminal_flag)

			if terminal_flag:
				self.train_on_memory()
				self.restart_game()
				return

	def train_on_memory(self):
		if len(self.player.memory) > 32:
			self.player.replay(batch_size=32)
			if self.n_games % 30 == 0:
				self.player.save_model()

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

		# paint inner trackpad
		qp.setPen(QPen(self.fishbowl_color, self.fishbowl_border_size, Qt.SolidLine))
		# draw fishbowl
		qp.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
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
		qp.drawEllipse(c + QPoint(*self.scale_point(np.squeeze(self.player.coords))), *([self.npc_pixel_radius] * 2))

	def animate_balls(self):
		self.update_thread = threading.Thread(target=self._animate_balls)
		self.update_thread.daemon = True
		self.update_thread.start()

	def _animate_balls(self):
		time.sleep(2)

		while True:
			self.animation_emitter.emit("act")
			time.sleep(0.001)
			self.animation_emitter.emit("learn")

	def restart_game(self):
		self.life_loop_counter = 0
		self.n_games += 1
		self.info_signal.emit("Game {0} - Exploration Rate {1}".format(self.n_games, self.player.epsilon))
		self.start_flag = True
		for enemy in self.enemies:
			enemy.revive()
		self.player.revive()


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
		self.layout.addWidget(self.n_games_label, 0, 0, 1, 10)
		self.fishbowl = Fishbowl(self.n_games_label.signal)
		self.layout.addWidget(self.fishbowl, 1, 0, 10, 10)

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
