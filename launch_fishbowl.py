from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QProgressBar, QComboBox, QDesktopWidget, \
	QGridLayout, QSlider, QGroupBox, QVBoxLayout, QHBoxLayout, QStyle, QScrollBar, QMainWindow, QAction, QDialog
from PyQt5.QtCore import QDateTime, Qt, QTimer, QPoint, pyqtSignal, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPalette
import time
import numpy as np
import threading
from collections import deque
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import random


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
			self.v = self.dir * np.random.randint(low=40, high=100, size=2) * 0.00002
			self.spherical_clip()

	@staticmethod
	def dist(a, b):
		return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	def revive(self):
		self.color = self.original_color
		self.coords = self.original_coords
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
		self.original_color = Qt.red
		self.color = Qt.red
		self.original_coords = np.array([0, 1.0])
		self.coords = self.original_coords


class Player(NPC):
	"""
	https://keon.io/deep-q-learning/
	"""

	def __init__(self, pixel_radius, fishbowl_pixel_radius, fishbowl_radius, state_size):
		super().__init__(pixel_radius, fishbowl_pixel_radius, fishbowl_radius)
		self.original_coords = np.array([0, -1.0])
		self.original_color = Qt.blue
		self.color = Qt.blue
		self.coords = self.original_coords
		self.step = 0.1
		self.grid_size = int(1 / self.step)

		# ----------------------------
		self.state_size = state_size
		self.action_size = 8
		self.memory = deque(maxlen=20000)
		self.gamma = 0.95  # discount rate
		self.epsilon = 0.5  # 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

		self.dist_history = []

	def _move(self, enemies):  # TODO apparently PEP8 does not let you change the args in an inhereted method...
		"""
		:param enemies: all enemy instances to be considered as inputs to determine where the layer shall move 
		"""

		enemies_coords = [x.coords if not x.dead else [0, 0] for x in enemies]

		# build current state
		state = self._build_state(enemies)
		# choose an action
		action = self.allowed_dirs[self.act(state)]
		# update player coords
		self.coords = self.coords + action * 0.002
		if not self.inside_sphere():
			self.spherical_clip()
		# compute reward
		reward = np.min([self.dist(self.coords, x.coords) for x in enemies])
		print(reward)
		# build next state
		next_state = self._build_state(enemies)
		# store state to memory
		self.remember(state, action, reward, next_state, False)

	def _build_state(self, enemies):
		enemies_grid_coords = [self._coords2grid(x.coords) for x in enemies]
		grid = np.zeros((self.grid_size*2, self.grid_size*2))
		for coords in enemies_grid_coords:
			grid[coords[0], coords[1]] += 1
		player_coords = self._coords2grid(self.coords)
		grid[player_coords[0], player_coords[1]] = -1.0

		return np.expand_dims(np.expand_dims(grid, axis=2), axis=0)

	def _coords2grid(self, coords):
		coords = (coords * (self.grid_size-1)).astype(int)
		coords[1] = -coords[1]
		coords += self.grid_size
		coords = coords[::-1]
		return coords

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(self.grid_size*2, self.grid_size*2, 1)))
		model.add(Conv2D(32, kernel_size=3, activation='relu'))
		model.add(Flatten())
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='softmax'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def revive(self):
		super().revive()
		self.memory.clear()

	def check_killed(self, enemies):
		for enemy in enemies:
			e = enemy.coords
			p = self.coords
			dist = self.dist(e, p)
			if dist < 2*self.r:  # two balls touching, 2 radius
				self.dead = True
				self.color = Qt.gray
				return True


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

		self.nenemies = 10
		self.enemies = [Enemy(
			self.npc_pixel_radius, self.fishbowl_pixel_radius, self.fishbowl_radius) for _ in range(self.nenemies)]
		self.player = Player(
			self.npc_pixel_radius, self.fishbowl_pixel_radius, self.fishbowl_radius, (self.nenemies + 1) * 2)

		self.n_games = 0
		self.info_signal = info_signal

	def scale_point(self, point):
		new_max = self.fishbowl_pixel_radius
		return ((p / self.fishbowl_radius) * new_max for p in point)

	def life_loop(self, command):
		"""
		:param command: unused
		this method imlements the main loop of the fishbowl
		in which we move around players, check who is dead and so on
		"""
		if self.player.check_killed(self.enemies):
			if len(self.player.memory) > 32:
				for i in range(10):
					self.info_signal.emit("Training batch {}".format(i))
					self.player.replay(32)
			self.restart_game()
			return

		self.player._move(self.enemies)

		for enemy in self.enemies:
			enemy.move()

		self.repaint()

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
		qp.setPen(QPen(self.fishbowl_color, self.fishbowl_thin_border_size, Qt.DashDotDotLine))
		for angle in range(0, 420, 45):
			line = QLineF(); line.setP1(c); line.setAngle(angle); line.setLength(self.fishbowl_pixel_radius)
			qp.drawLine(line)
		# draw wheel separators
		line = QLineF(); line.setP1(c + QPoint(self.wheel_size, 0)); line.setAngle(0); line.setLength(self.wheel_width)
		qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(0, -self.wheel_size)); line.setAngle(90); line.setLength(self.wheel_width)
		qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(-self.wheel_size, 0)); line.setAngle(180); line.setLength(self.wheel_width)
		qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(0, self.wheel_size)); line.setAngle(270); line.setLength(self.wheel_width)
		qp.drawLine(line)

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
			time.sleep(0.00001)
			self.animation_emitter.emit("animate")

	def restart_game(self):
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
