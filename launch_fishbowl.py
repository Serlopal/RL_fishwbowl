from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QProgressBar, QComboBox, QDesktopWidget, \
	QGridLayout, QSlider, QGroupBox, QVBoxLayout, QHBoxLayout, QStyle, QScrollBar, QMainWindow, QAction, QDialog
from PyQt5.QtCore import QDateTime, Qt, QTimer, QPoint, pyqtSignal, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPalette
import time
import numpy as np
import threading

class SignalEmitter(QWidget):
	valueChanged = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		self._value = 0

	@property
	def value(self):
		return self._t

	@value.setter
	def value(self, value):
		self._value = value
		self.valueChanged.emit(value)


class MushroomWidget(QWidget):
	def __init__(self):
		super().__init__()

		self.origin = np.array([0, 0])
		self.nenemies = 10
		self.nplayers = 1

		self.balls_coords_emitter = SignalEmitter()
		self.internal_balls_coords = np.zeros((self.nenemies + self.nplayers, 2))
		self.enemies_color = Qt.red

		self.players_color = Qt.blue

		self.origin_color = Qt.black
		self.mushroom_color = Qt.black

		self.mushroom_size_origin = 350
		self.mushroom_size = self.mushroom_size_origin
		self.wheel_size = self.mushroom_size_origin * 1.15
		self.wheel_width = self.mushroom_size_origin * 0.15
		self.cursor_size = self.mushroom_size_origin * 0.05
		self.mushroom_border_size = self.mushroom_size_origin * 0.004
		self.mushroom_thin_border_size = self.mushroom_size_origin * 0.002


		self.coords_emitter = SignalEmitter()
		# connect signal from emitter to change value of button
		self.balls_coords_emitter.valueChanged.connect(lambda value: self.setCoords(value))

	def scale_point(self, point):
		original_max = 0.5
		new_max = self.mushroom_size
		return ((p / original_max) * new_max for p in point)

	def setCoords(self, value):
		self.internal_balls_coords = value
		self.repaint()

	def setTap(self, tap):
		if tap:
			self.mushroom_size = self.mushroom_size * 0.9
		else:
			self.mushroom_size = self.mushroom_size_origin
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
		qp.setPen(QPen(self.mushroom_color, self.mushroom_border_size, Qt.SolidLine))
		# draw mushroom
		qp.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
		qp.drawEllipse(c, *([self.mushroom_size]*2))

		# draw axis lines
		qp.setPen(QPen(self.mushroom_color, self.mushroom_thin_border_size, Qt.DashDotDotLine))
		for angle in range(0, 420, 45):
			line = QLineF(); line.setP1(c); line.setAngle(angle); line.setLength(self.mushroom_size)
			qp.drawLine(line)
		# draw wheel separators
		line = QLineF(); line.setP1(c + QPoint(self.wheel_size, 0)); line.setAngle(0); line.setLength(self.wheel_width);qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(0, -self.wheel_size)); line.setAngle(90); line.setLength(self.wheel_width);qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(-self.wheel_size, 0)); line.setAngle(180); line.setLength(self.wheel_width);qp.drawLine(line)
		line = QLineF(); line.setP1(c + QPoint(0, self.wheel_size)); line.setAngle(270); line.setLength(self.wheel_width);qp.drawLine(line)

		# draw enemies
		for i, ball in enumerate(self.internal_balls_coords):
			qp.setPen(QPen(self.mushroom_color, self.mushroom_border_size, Qt.SolidLine))
			if i < self.nenemies:
				qp.setBrush(QBrush(self.enemies_color, Qt.SolidPattern))
			else:
				qp.setBrush(QBrush(self.players_color, Qt.SolidPattern))

			qp.drawEllipse(c + QPoint(*self.scale_point(ball)), *([self.cursor_size] * 2))

	def animate_balls(self):
		self.update_thread = threading.Thread(target=self._animate_balls)
		self.update_thread.daemon = True
		self.update_thread.start()

	def _animate_balls(self):
		nballs = self.nenemies + self.nplayers
		balls = np.zeros((nballs, 2))
		v_prev_dir = np.empty((nballs, 2))
		v_dir = np.empty((nballs , 2))
		v_speed = np.empty((nballs, 2))
		v = np.zeros((nballs, 2))
		first = np.ones((nballs, 1))

		while True:
			for i in range(len(balls)):
				balls[i] += v[i]
				if not self.inside_sphere(balls[i]) or first[i]:
					first[i] = 0
					while True:
						v_dir[i] = np.random.choice([1, 0, -1], size=2, replace=True)
						if np.any(v_dir[i] != v_prev_dir[i]) and np.any(v_dir[i]):
							v_prev_dir[i] = v_dir[i]
							break
					v_speed[i] = np.random.randint(low=10, high=20, size=2) * 0.0001
					v[i] = np.multiply(v_speed[i], v_dir[i])  # np.random.randint(low=10, high=20) * 0.0001

					balls[i] = self.spherical_clip(balls[i])
			time.sleep(0.01)
			self.balls_coords_emitter.value = balls

	def check_collisions(self):
		self.collision_thread = threading.Thread(target=self._check_collision)
		self.collision_thread.daemon = True
		self.collision_thread.start()

	def _check_collision(self):
		time.sleep(2)
		enemies = self.internal_balls_coords[0:self.nenemies]
		players = self.internal_balls_coords[self.nenemies:]

		while True:
			for p in players:
				for e in enemies:
					dist, d = np.sqrt((e[0] - p[0]) ** 2 + (e[1] - p[1]) ** 2) , (self.cursor_size / self.mushroom_size)
					if dist < d:
						print("DEAD")
						break
	@staticmethod
	def spherical_clip(p, r=0.47):
		p = np.array(p)
		dist = np.sqrt(np.sum(p ** 2))
		if dist > r:
			p = p * (r / dist)
		return p

	@staticmethod
	def inside_sphere(p, r=0.46):
		dist = np.sqrt(np.sum(np.array(p) ** 2))
		if dist > r:
			return False
		else:
			return True


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
		self.window.move(ag.width()*0.15, ag.height()*0.05)
		self.window.setMinimumWidth(ag.width()*0.3)
		self.window.setMinimumHeight(ag.height()*0.4)

		self.layout =QVBoxLayout()
		self.create_mushroom_group()
		self.layout.addWidget(self.mushroom_group)
		self.main_group.setLayout(self.layout)

		# set layout inside window
		self.window.setLayout(self.layout)
		self.window.show()

		self.start_animation()

	def create_mushroom_group(self):
		# create group
		self.mushroom_group = QGroupBox("Mushroom")
		# create widgets
		self.mushroom = MushroomWidget()
		# create group layout
		layout = QVBoxLayout()
		layout.addWidget(self.mushroom)
		# add layout to group
		self.mushroom_group.setLayout(layout)

	def start_UI(self):
		self.app.exec()

	def start_animation(self):
		time.sleep(1)
		self.mushroom.animate_balls()
		self.mushroom.check_collisions()







if __name__ == "__main__":

	ui = gameUI()
	ui.start_UI()
