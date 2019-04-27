from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QProgressBar, QComboBox, QDesktopWidget, \
	QGridLayout, QSlider, QGroupBox, QVBoxLayout, QHBoxLayout, QStyle, QScrollBar, QMainWindow, QAction, QDialog
from PyQt5.QtCore import QDateTime, Qt, QTimer, QPoint, pyqtSignal, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPalette, QScreen, QImage, QGuiApplication
import time
import numpy as np
import threading
from collections import deque
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, Conv3D
from keras.optimizers import Adam
from keras.models import load_model
import random
import os
from cv2 import resize, INTER_CUBIC, imwrite

import copy
import numpy as np


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
	allowed_dirs = np.array([np.array(x) for x in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]])

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius):
		self.original_coords = np.array([0, 0])
		self.r = pixel_radius / fishbowl_pixel_radius
		self.fishbowl_radius = fishbowl_radius
		self.original_color = Qt.black
		self.color = Qt.black
		self.coords = self.original_coords
		self.v = np.array([0, 0])
		self.speed = 0.001
		self.pvdi = np.array([0, 0])
		self.first = True
		self.dead = False

	def move(self):
		self.coords = self.coords + self.v
		if not self.inside_sphere() or self.first:
			if not self.first:
				forbidden_dirs = [self.pvdi - 1 if self.pvdi != 0 else len(self.allowed_dirs) - 1,
								  self.pvdi,
								  self.pvdi + 1 if self.pvdi != len(self.allowed_dirs) - 1 else 0]
				available_dirs = np.delete(self.allowed_dirs, forbidden_dirs, axis=0)
				chosen_dir = np.random.randint(low=0, high=len(available_dirs))
				self.dir = available_dirs[chosen_dir, :]
				self.pvdi = np.where(np.all(self.allowed_dirs == self.dir, axis=1))[0][0]
			else:
				chosen_dir = np.random.randint(low=0, high=len(self.allowed_dirs))
				self.dir = self.allowed_dirs[chosen_dir, :]
				self.pvdi = chosen_dir
			self.first = False
			self.v = self.dir * np.random.randint(low=40, high=100, size=2) * self.speed
			self.spherical_clip()

	@staticmethod
	def dist(a, b):
		return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	def revive(self):
		self.color = self.original_color
		self.coords = (np.array([np.random.rand(), np.random.rand()]) * 2) - 1
		self.spherical_clip()
		self.v = np.array([0, 0])
		self.first = True
		self.dead = False

	def spherical_clip(self):
		dist = np.sqrt(np.sum(self.coords ** 2))
		if dist > self.fishbowl_radius - self.r:
			self.coords = self.coords * ((self.fishbowl_radius - self.r) / dist)

	def inside_sphere(self):
		dist = np.sqrt(np.sum(self.coords ** 2))
		if dist > (self.fishbowl_radius - self.r):
			return False
		else:
			return True


class Enemy(NPC):

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.original_color = Qt.white
		self.color = self.original_color
		self.original_coords = np.array([np.random.rand(), np.random.rand()])
		self.spherical_clip()
		self.coords = self.original_coords


class Player(NPC):
	"""
	https://keon.io/deep-q-learning/
	"""

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, state_size):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.allowed_dirs = np.vstack([self.allowed_dirs, np.reshape([0, 0], (1, 2))])

		self.original_coords = (np.array([np.random.rand(), np.random.rand()]) * 2) - 1
		self.spherical_clip()
		self.original_color = Qt.black
		self.color = self.original_color
		self.coords = self.original_coords
		self.frame_size = 32

		# ----------------------------
		self.state_size = state_size
		self.action_size = 9
		self.memory = deque(maxlen=20000)
		self.gamma = 0.95  # discount rate
		self.epsilon = 1.0  # 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.curr_state = None
		self.curr_action = None
		self.speed = 0.01
		# self.learning_rate = 0.001
		self.wlen = 4
		self.frame_memory = deque(maxlen=self.wlen)
		self.model_name = "model.h5"
		self.model = self._build_model()

	def act(self):  # TODO apparently PEP8 does not let you change the args in an inhereted method...
		"""
		:param frame: all enemy instances to be considered as inputs to determine where the layer shall move
		"""

		# build current state
		state = self._build_state()

		# choose an action
		action = self.allowed_dirs[self.choose_action(state)]
		# update player coords
		self.coords = self.coords + action * self.speed
		if not self.inside_sphere():
			self.spherical_clip()

		# store state
		self.curr_state = state
		self.curr_action = action

	def remember(self, terminal_flag, enemies):
		# build next state
		next_state = self._build_state()
		# compute reward and check if state is terminal
		reward = min([self.dist(self.coords, x.coords) for x in enemies]) if not terminal_flag else 0.0
		# reward = 1.0 if not terminal_flag else -10.0

		# store state to memory
		self.save_to_memory(self.curr_state, self.curr_action, reward, next_state, terminal_flag)

	def _build_state(self):
		# now build cube
		stacked_memory = np.stack(self.frame_memory, axis=2)
		return np.expand_dims(stacked_memory, axis=0)

	def _coords2grid(self, coords):
		coords = (coords * (self.frame_size - 1)).astype(int)
		coords[1] = -coords[1]
		coords += self.frame_size
		coords = coords[::-1]
		return coords

	def _build_model(self):
		if os.path.exists("model.h5"):
			print("found previous model	")
			model = load_model(self.model_name)
			return model
		else:
			# Neural Net for Deep-Q learning Model
			model = Sequential()
			model.add(Conv2D(64, kernel_size=5, activation='relu'))
			model.add(Conv2D(32, kernel_size=3, activation='relu'))
			model.add(Flatten())
			model.add(Dense(32, activation='relu'))
			model.add(Dropout(0.1))
			model.add(Dense(self.action_size, activation='linear'))
			model.compile(loss='mse', optimizer=Adam())
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

	def replay(self, batch_size, num_batches):
		for _ in range(num_batches):
			minibatch = random.sample(self.memory, batch_size)
			for state, action, reward, next_state, done in minibatch:
				if done:
					target = reward
				else:
					target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
				target_f = self.model.predict(state)
				target_f[0][action] = target
				self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def revive(self):
		super().revive()
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

		self.fishbowl_color = Qt.black
		self.fishbowl_radius = 1.0
		self.fishbowl_pixel_radius = 100
		self.wheel_size = self.fishbowl_pixel_radius * 1.15
		self.wheel_width = self.fishbowl_pixel_radius * 0.15
		self.npc_pixel_radius = self.fishbowl_pixel_radius * 0.1
		self.fishbowl_border_size = self.fishbowl_pixel_radius * 0.004
		self.fishbowl_thin_border_size = self.fishbowl_pixel_radius * 0.002

		self.nenemies = 3
		self.enemies = [Enemy(
			self.npc_pixel_radius, self.fishbowl_pixel_radius, self.fishbowl_radius) for _ in range(self.nenemies)]

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
			# append at least one frame, or fill with the same in the beginning to build a full state
			while True:
				frame = self.player.proccess_frame(self.get_frame())
				self.player.frame_memory.append(frame)
				if len(self.player.frame_memory) >= self.player.wlen:
					break

			# issue player action
			self.player.act()
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

			# store action knowledge and check if the state is terminal
			terminal_flag = self.player.check_killed(enemies=self.enemies)

			self.player.remember(terminal_flag, self.enemies)

			if terminal_flag:
				self.train_on_memory()
				self.restart_game()
				return

	def train_on_memory(self):
		if len(self.player.memory) > 32:
			self.player.replay(batch_size=32, num_batches=1)
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
			qp.drawEllipse(c + QPoint(*self.scale_point(enemy.coords)), *([self.npc_pixel_radius] * 2))

		# draw player
		qp.setBrush(QBrush(self.player.color, Qt.SolidPattern))
		qp.drawEllipse(c + QPoint(*self.scale_point(self.player.coords)), *([self.npc_pixel_radius] * 2))

	def animate_balls(self):
		self.update_thread = threading.Thread(target=self._animate_balls)
		self.update_thread.daemon = True
		self.update_thread.start()

	def _animate_balls(self):
		while True:
			self.animation_emitter.emit("act")
			time.sleep(0.00001)
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
		time.sleep(1)
		self.fishbowl.animate_balls()


if __name__ == "__main__":

	ui = gameUI()
	ui.start_ui()
