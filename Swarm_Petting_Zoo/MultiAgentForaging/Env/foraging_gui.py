from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QPushButton
from PyQt5.QtCore import Qt
from simulation_runner import run_simulation  # Import the simulation function
from foraging_config import ForagingConfig
import sys

class ConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.config = ForagingConfig()  # Default config
        self.simulation_running = False
        self.simulation_paused = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Foraging Simulation Config")

        # Create a layout
        layout = QVBoxLayout(self)

        # SpinBox for number of agents
        self.num_agents_spinbox = QSpinBox()
        self.num_agents_spinbox.setRange(1, 20)
        self.num_agents_spinbox.setValue(self.config.num_agents)
        self.num_agents_spinbox.valueChanged.connect(self.update_num_agents)
        layout.addWidget(QLabel('Number of Agents'))
        layout.addWidget(self.num_agents_spinbox)

        # Slider for field of view (FOV)
        self.fov_slider = QSlider(Qt.Horizontal)
        self.fov_slider.setRange(1, 5)
        self.fov_slider.setValue(self.config.fov)
        self.fov_slider.valueChanged.connect(self.update_fov)
        layout.addWidget(QLabel('Field of View (FOV)'))
        layout.addWidget(self.fov_slider)

        # SpinBox for resource reward
        self.resource_reward_spinbox = QSpinBox()
        self.resource_reward_spinbox.setRange(0, 1000)
        self.resource_reward_spinbox.setValue(self.config.resource_reward)
        self.resource_reward_spinbox.valueChanged.connect(self.update_resource_reward)
        layout.addWidget(QLabel('Resource Reward'))
        layout.addWidget(self.resource_reward_spinbox)

        # SpinBox for battery usage rate
        self.battery_usage_spinbox = QSpinBox()
        self.battery_usage_spinbox.setRange(0, 100)
        self.battery_usage_spinbox.setValue(self.config.battery_usage_rate)
        self.battery_usage_spinbox.valueChanged.connect(self.update_battery_usage_rate)
        layout.addWidget(QLabel('Battery Usage Rate'))
        layout.addWidget(self.battery_usage_spinbox)

        # SpinBox for battery charge cost
        self.battery_charge_cost_spinbox = QSpinBox()
        self.battery_charge_cost_spinbox.setRange(0, 100)
        self.battery_charge_cost_spinbox.setValue(self.config.battery_charge_cost)
        self.battery_charge_cost_spinbox.valueChanged.connect(self.update_battery_charge_cost)
        layout.addWidget(QLabel('Battery Charge Cost'))
        layout.addWidget(self.battery_charge_cost_spinbox)

        # SpinBox for battery charge amount
        self.battery_charge_amount_spinbox = QSpinBox()
        self.battery_charge_amount_spinbox.setRange(0, 100)
        self.battery_charge_amount_spinbox.setValue(self.config.battery_charge_amount)
        self.battery_charge_amount_spinbox.valueChanged.connect(self.update_battery_charge_amount)
        layout.addWidget(QLabel('Battery Charge Amount'))
        layout.addWidget(self.battery_charge_amount_spinbox)

        # SpinBox for minimum battery level
        self.min_battery_level_spinbox = QSpinBox()
        self.min_battery_level_spinbox.setRange(0, 100)
        self.min_battery_level_spinbox.setValue(self.config.min_battery_level)
        self.min_battery_level_spinbox.valueChanged.connect(self.update_min_battery_level)
        layout.addWidget(QLabel('Minimum Battery Level'))
        layout.addWidget(self.min_battery_level_spinbox)

        # DoubleSpinBox for battery recharge threshold
        self.battery_recharge_threshold_spinbox = QDoubleSpinBox()
        self.battery_recharge_threshold_spinbox.setRange(0.0, 1.0)
        self.battery_recharge_threshold_spinbox.setSingleStep(0.01)
        self.battery_recharge_threshold_spinbox.setValue(self.config.battery_recharge_threshold)
        self.battery_recharge_threshold_spinbox.valueChanged.connect(self.update_battery_recharge_threshold)
        layout.addWidget(QLabel('Battery Recharge Threshold'))
        layout.addWidget(self.battery_recharge_threshold_spinbox)

        # Button to start the simulation
        start_button = QPushButton("Start Simulation")
        start_button.clicked.connect(self.start_simulation)
        layout.addWidget(start_button)
        
        # Button to pause the simulation
        pause_button = QPushButton("Pause Simulation")
        pause_button.clicked.connect(self.pause_simulation)
        layout.addWidget(pause_button)

        # Button to exit the simulation
        exit_button = QPushButton("Exit Simulation")
        exit_button.clicked.connect(self.exit_simulation)
        layout.addWidget(exit_button)

        # Set the layout for the widget
        self.setLayout(layout)
        self.show()

    def process_events(self):
        """Process the GUI events to keep the interface responsive."""
        QApplication.processEvents()

    def update_num_agents(self, value):
        self.config.num_agents = value

    def update_fov(self, value):
        self.config.fov = value

    def update_resource_reward(self, value):
        self.config.resource_reward = value

    def update_battery_usage_rate(self, value):
        self.config.battery_usage_rate = value

    def update_battery_charge_cost(self, value):
        self.config.battery_charge_cost = value

    def update_battery_charge_amount(self, value):
        self.config.battery_charge_amount = value

    def update_min_battery_level(self, value):
        self.config.min_battery_level = value

    def update_battery_recharge_threshold(self, value):
        self.config.battery_recharge_threshold = value

    def start_simulation(self):
        if not self.simulation_running:
            self.simulation_running = True
            run_simulation(config=self.config, gui=self)
            self.simulation_running = False
        
    def pause_simulation(self):
        if self.simulation_running:
            self.simulation_paused = not self.simulation_paused

    def exit_simulation(self):
        if self.simulation_running:
            self.simulation_running = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ConfigGUI()
    gui.show()
    sys.exit(app.exec_())
