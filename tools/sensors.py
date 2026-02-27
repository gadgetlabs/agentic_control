"""
tools/sensors.py
Sensor read tool for the Strands PlanningAgent.

robot_state is populated at 10 Hz by serial_reader.py from the Arduino's
$IMU / $CMP / $ODO / $RPM / $LDR telemetry lines.
"""

from strands import tool

from serial_reader import robot_state


@tool
def get_sensors() -> dict:
    """
    Return the latest sensor readings from the robot:
    IMU (accel + gyro), compass, odometry (linear/angular velocity), wheel RPM.
    """
    return robot_state
