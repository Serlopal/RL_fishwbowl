from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QProgressBar, QComboBox, QDesktopWidget, \
	QGridLayout, QSlider, QGroupBox, QVBoxLayout, QHBoxLayout, QStyle, QScrollBar, QMainWindow, QAction, QDialog
from PyQt5.QtCore import QDateTime, Qt, QTimer, QPoint, pyqtSignal, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPalette
import time
import numpy as np
import threading
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random


class DynamicLabel(QLabel):
	signal = pyqtSignal(object)

	def __init__(self, base_text):
		super().__init__()
		self.font = QFont("Times", 30, QFont.Bold)
		self.setFont(self.font)

		self.base_text = base_text
		self.setText(self.base_text + "0")
		self.signal.connect(lambda x: self.setText(self.base_text + x))


class NPC():
	allowed_dirs = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])

	def __init__(self, diameter, fishbowl_diameter):
		self.d = diameter / fishbowl_diameter
		self.original_color = Qt.black
		self.color = Qt.black
		self.coords = np.array([0, 0])
		self.v = np.array([0, 0])
		self.pvdi = np.array([0, 0])
		self.first = True
		self.dead = False

	def move(self):
		if not self.dead:
			self.coords = self.coords + self.v
			if not self.inside_sphere(self.coords) or self.first:
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
				self.coords = self.spherical_clip(self.coords)

	def check_killed_by(self, player):
		p = player.coords
		e = self.coords
		dist = self.dist(e, p)
		if dist < self.d:
			self.dead = True
			self.color = Qt.gray

	@staticmethod
	def dist(a, b):
		return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	def revive(self):
		self.color = self.original_color
		self.coords = np.array([0, 0])
		self.v = np.array([0, 0])
		self.first = True
		self.dead = False

	def spherical_clip(self, p, r=0.499):
		r -= self.d/2
		p = np.array(p)
		dist = np.sqrt(np.sum(p ** 2))
		if dist > r:
			p = p * (r / dist)
		return p

	def inside_sphere(self, p, r=0.50):
		r -= self.d/2
		dist = np.sqrt(np.sum(np.array(p) ** 2))
		if dist > r:
			return False
		else:
			return True


class Enemy(NPC):
	def __init__(self, diameter, fishbowl_diameter):
		super().__init__(diameter, fishbowl_diameter)
		self.original_color = Qt.red
		self.color = Qt.red


class Player(NPC):
	"""
	https://keon.io/deep-q-learning/
	"""
	def __init__(self, diameter, fishbowl_diameter, state_size):
		super().__init__(diameter, fishbowl_diameter)
		self.original_color = Qt.blue
		self.color = Qt.blue

		# ----------------------------
		self.state_size = state_size
		self.action_size = 2 # we move in a 2D world
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95  # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def move(self, enemies_coords): # TODO apparently PEP8 does not let you change the args in an inhereted method...
		# build current state
		state = np.concatenate([enemies_coords, np.reshape(self.coords, (1, -1))], axis=0)
		state = np.reshape(state, (1, -1))
		# choose an action
		action = np.squeeze(self.model.predict(state))
		# update player coords
		self.coords = self.coords + (action - 1) * 0.002
		if not self.inside_sphere(self.coords):
			self.coords = self.spherical_clip(self.coords)
		# compute reward
		reward = 1 / np.min([self.dist(self.coords, x) for x in enemies_coords])
		# print(reward)
		# build next state
		next_state = np.reshape(np.concatenate([enemies_coords, np.reshape(self.coords, (1, -1))], axis=0), (1, -1))
		# store state to memory
		self.remember(state, action, reward, next_state, False)

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='sigmoid'))
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


class Fishbowl(QWidget):
	animation_emitter = pyqtSignal(object)

	def __init__(self, n_games_signal):
		super().__init__()

		self.render_bowl = True

		# connect signal from emitter to trigger the animation
		self.animation_emitter.connect(lambda x: self._move_npcs(x))

		self.fishbowl_color = Qt.black

		self.fishbowl_size = 350
		self.wheel_size = self.fishbowl_size * 1.15
		self.wheel_width = self.fishbowl_size * 0.15
		self.npc_size = self.fishbowl_size * 0.3
		self.fishbowl_border_size = self.fishbowl_size * 0.004
		self.fishbowl_thin_border_size = self.fishbowl_size * 0.002

		self.nenemies = 10
		self.enemies = [Enemy(self.npc_size, self.fishbowl_size) for _ in range(self.nenemies)]
		self.player = Player(self.npc_size, self.fishbowl_size, (self.nenemies + 1) * 2)

		self.start_flag = True
		self.start_time = None
		self.time_advantage = 2
		self.n_games = 0
		self.n_games_signal = n_games_signal

	def scale_point(self, point):
		original_max = 0.5
		new_max = self.fishbowl_size
		return ((p / original_max) * new_max for p in point)

	def _move_npcs(self, command):
		# start moving enemies but not player yet
		if self.start_flag:
			self.start_flag = False
			self.start_time = time.time()
		go_player = time.time() - self.start_time > self.time_advantage

		# check someone is still alive
		alive_enemies = [x for x in self.enemies if not x.dead]
		if not alive_enemies:
			if len(self.player.memory) > 32:
				self.player.replay(32)
			self.restart_game()
			return

		for enemy in alive_enemies:
			enemy.move()
			# check dead
			if go_player:
				enemy.check_killed_by(self.player)
		if go_player:
			pass
			self.player.move([x.coords if not x.dead else [-1, -1] for x in self.enemies])

		if self.render_bowl:
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
		qp.drawEllipse(c, *([self.fishbowl_size] * 2))

		# draw axis lines
		qp.setPen(QPen(self.fishbowl_color, self.fishbowl_thin_border_size, Qt.DashDotDotLine))
		for angle in range(0, 420, 45):
			line = QLineF(); line.setP1(c); line.setAngle(angle); line.setLength(self.fishbowl_size)
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

		# draw dead enemies
		for i, enemy in enumerate([x for x in self.enemies if x.dead]):
			qp.setBrush(QBrush(enemy.color, Qt.SolidPattern))
			qp.drawEllipse(c + QPoint(*self.scale_point(enemy.coords)), *([self.npc_size] * 2))
		# draw alive enemies
		for i, enemy in enumerate([x for x in self.enemies if not x.dead]):
			qp.setBrush(QBrush(enemy.color, Qt.SolidPattern))
			qp.drawEllipse(c + QPoint(*self.scale_point(enemy.coords)), *([self.npc_size] * 2))

		# draw player
		qp.setBrush(QBrush(self.player.color, Qt.SolidPattern))
		qp.drawEllipse(c + QPoint(*self.scale_point(self.player.coords)), *([self.npc_size] * 2))

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
		self.n_games_signal.emit(str(self.n_games))
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
		self.layout.addWidget(self.n_games_label, 0,0,1,10)
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
