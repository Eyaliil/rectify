#!/usr/bin/env python3
"""
Flask backend for FlexTail sensor web interface.
Streams sensor data via Socket.IO to connected clients.
"""

import asyncio
import sys
import math
import os
from pathlib import Path
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading



from fiffi_unleashed import FlexTailApp, HardwareVersion
import flexlib as fl

# AI model integration
try:
    from ml.inference import create_classifier
    ai_classifier = create_classifier()
    print("✓ AI classifier initialized")
except Exception as e:
    print(f"⚠ AI classifier initialization failed: {e}")
    print("  Continuing without AI features")
    ai_classifier = None


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
sensor_app = None
sensor_task = None
stop_event = threading.Event()
measurement_count = 0
ai_enabled = False  # Toggle for AI classification


class AsyncLoopThread(threading.Thread):
    """Dedicated thread for running the asyncio event loop."""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = None
        self.ready = threading.Event()
        
    def run(self):
        """Run the event loop in this thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()
        
    def run_coroutine(self, coro):
        """Schedule a coroutine to run in the loop and return a Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
        
    def stop(self):
        """Stop the event loop."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# Global async loop thread
async_loop_thread = AsyncLoopThread()
async_loop_thread.start()
async_loop_thread.ready.wait()  # Wait for loop to be ready


class SensorManager:
    """Manages the FlexTail sensor connection and data streaming."""
    
    def __init__(self):
        self.app = None
        self.connected = False
        self.streaming = False
        self.mac_address = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 3
        
    def on_measurement(self, measurement: fl.Measurement):
        """Called when a measurement is received from the sensor."""
        global ai_enabled, ai_classifier

        try:
            # Calculate orientation from accelerometer
            orientation = fl.CaseOrientation.from_acc(measurement.acc)

            # Get angles
            angles = measurement.angles

            # Convert radians to degrees for display
            bend_deg = angles.bend * 180 / math.pi
            pitch_deg = orientation.pitch * 180 / math.pi
            roll_deg = orientation.roll * 180 / math.pi

            # Convert measurement to dict for JSON serialization (for UI)
            data = {
                'timestamp': measurement.timestamp,
                'bend': bend_deg,
                'pitch': pitch_deg,
                'roll': roll_deg,
                'raw_data': str(measurement)
            }

            # Emit raw measurement data
            socketio.emit('measurement_data', data)

            # AI Classification
            if ai_enabled and ai_classifier and ai_classifier.is_loaded:
                # Extract ALL 8 features for the NEW model
                # Calculate acceleration magnitude (acc is a list [x, y, z])
                acc_mag = 0.0
                if hasattr(measurement, 'acc') and isinstance(measurement.acc, list) and len(measurement.acc) >= 3:
                    acc_mag = (measurement.acc[0]**2 + measurement.acc[1]**2 + measurement.acc[2]**2)**0.5

                # Calculate gyro magnitude (gyro is a list [x, y, z])
                gyro_mag = 0.0
                if hasattr(measurement, 'gyro') and measurement.gyro and isinstance(measurement.gyro, list) and len(measurement.gyro) >= 3:
                    gyro_mag = (measurement.gyro[0]**2 + measurement.gyro[1]**2 + measurement.gyro[2]**2)**0.5

                ai_data = {
                    'timestamp': measurement.timestamp,
                    'lumbarAngle': angles.bend,
                    'twist': angles.twist,
                    'lateral': measurement.lateral_flexion if hasattr(measurement, 'lateral_flexion') else 0.0,
                    'sagittal': measurement.sagittal_flexion if hasattr(measurement, 'sagittal_flexion') else 0.0,
                    'lateralApprox': measurement.calc_lateral_approx() if hasattr(measurement, 'calc_lateral_approx') else 0.0,
                    'sagittalApprox': measurement.calc_sagittal_approx() if hasattr(measurement, 'calc_sagittal_approx') else 0.0,
                    'acceleration': acc_mag,
                    'gyro': gyro_mag
                }

                # Add measurement to classifier buffer
                ready = ai_classifier.add_measurement(ai_data)

                # Emit buffer status
                buffer_status = ai_classifier.get_buffer_status()
                socketio.emit('ai_buffer_status', buffer_status)

                # Perform prediction when buffer is ready
                if ready:
                    prediction = ai_classifier.predict()
                    if prediction:
                        socketio.emit('ai_classification', prediction)

            self.reconnect_attempts = 0  # Reset on successful data
        except Exception as e:
            print(f"Error processing measurement: {e}")
            socketio.emit('error', {'message': f'Measurement error: {str(e)}'})
    
    async def connect_to_sensor(self, mac_address=None, hw_version='6.0'):
        """Connect to the FlexTail sensor with reconnection logic."""
        self.mac_address = mac_address
        
        while self.reconnect_attempts < self.max_reconnect_attempts and not stop_event.is_set():
            try:
                self.reconnect_attempts += 1
                socketio.emit('connection_status', {
                    'status': 'connecting',
                    'attempt': self.reconnect_attempts,
                    'max_attempts': self.max_reconnect_attempts
                })
                
                # Create new app instance
                self.app = FlexTailApp()
                
                # Configure hardware version
                hw_ver = HardwareVersion.HW6_0 if hw_version == '6.0' else HardwareVersion.HW5_0
                self.app.set_config(hw_version=hw_ver, use_serial=False)
                
                # Set MAC address filter if provided
                if mac_address:
                    try:
                        self.app.set_manufacturer_id_from_string(mac_address)
                        print(f"Connecting to device: {mac_address}")
                    except ValueError as e:
                        socketio.emit('error', {'message': f'Invalid MAC address: {str(e)}'})
                        return False
                
                # Add measurement listener
                self.app.add_measurement_listener(self.on_measurement)
                
                # Attempt connection
                if await self.app.connect():
                    self.connected = True
                    self.reconnect_attempts = 0
                    socketio.emit('connection_status', {'status': 'connected'})
                    print("Connected to sensor!")
                    return True
                else:
                    print(f"Connection attempt {self.reconnect_attempts} failed")
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        socketio.emit('connection_status', {
                            'status': 'retrying',
                            'delay': self.reconnect_delay
                        })
                        await asyncio.sleep(self.reconnect_delay)
                        
            except Exception as e:
                print(f"Connection error: {e}")
                socketio.emit('error', {'message': f'Connection error: {str(e)}'})
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
        
        socketio.emit('connection_status', {'status': 'failed'})
        return False
    
    async def start_measurement(self, frequency=25):
        """Start streaming measurements from the sensor."""
        if not self.connected or not self.app:
            socketio.emit('error', {'message': 'Not connected to sensor'})
            return False
        
        try:
            # Set frequency
            await self.app.set_frequency(frequency)
            print(f"Frequency set to {frequency}Hz")
            
            # Start measurement
            await self.app.start_measurement()
            self.streaming = True
            socketio.emit('streaming_status', {'streaming': True})
            print("Measurement stream started")
            
            # Keep the connection alive and monitor
            await self.monitor_connection()
            
            return True
        except Exception as e:
            print(f"Error starting measurement: {e}")
            socketio.emit('error', {'message': f'Error starting measurement: {str(e)}'})
            return False
    
    async def monitor_connection(self):
        """Monitor connection and attempt reconnection if lost."""
        while self.streaming and not stop_event.is_set():
            await asyncio.sleep(1)
            
            # If connection is lost, attempt to reconnect
            if not self.connected:
                print("Connection lost, attempting to reconnect...")
                socketio.emit('connection_status', {'status': 'reconnecting'})
                if await self.connect_to_sensor(self.mac_address):
                    # Restart measurement after reconnection
                    await self.app.start_measurement()
    
    async def stop_measurement(self):
        """Stop streaming measurements."""
        if self.app and self.streaming:
            try:
                await self.app.stop_measurement()
                self.streaming = False
                socketio.emit('streaming_status', {'streaming': False})
                print("Measurement stream stopped")
            except Exception as e:
                print(f"Error stopping measurement: {e}")
    
    async def disconnect(self):
        """Disconnect from the sensor."""
        await self.stop_measurement()
        if self.app:
            try:
                await self.app.disconnect()
                self.connected = False
                socketio.emit('connection_status', {'status': 'disconnected'})
                print("Disconnected from sensor")
            except Exception as e:
                print(f"Error disconnecting: {e}")


sensor_manager = SensorManager()


@app.route('/')
def index():
    """Health-check endpoint for deployment orchestration."""
    return jsonify({
        'status': 'ok',
        'service': 'FlexTail Sensor Backend',
        'streaming': sensor_manager.streaming,
        'connected': sensor_manager.connected
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connection_status', {'status': 'ready'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('connect_sensor')
def handle_connect_sensor(data):
    """Handle sensor connection request."""
    mac_address = data.get('mac_address')
    hw_version = data.get('hw_version', '6.0')
    
    print(f"Connection request: MAC={mac_address}, HW={hw_version}")
    
    # Run connection using the persistent event loop
    async_loop_thread.run_coroutine(
        sensor_manager.connect_to_sensor(mac_address, hw_version)
    )


@socketio.on('start_measurement')
def handle_start_measurement(data):
    """Handle start measurement request."""
    frequency = data.get('frequency', 25)
    
    print(f"Start measurement request: {frequency}Hz")
    
    # Run measurement using the persistent event loop
    async_loop_thread.run_coroutine(
        sensor_manager.start_measurement(frequency)
    )


@socketio.on('stop_measurement')
def handle_stop_measurement():
    """Handle stop measurement request."""
    print("Stop measurement request")
    
    # Run stop using the persistent event loop
    async_loop_thread.run_coroutine(
        sensor_manager.stop_measurement()
    )


@socketio.on('disconnect_sensor')
def handle_disconnect_sensor():
    """Handle sensor disconnection request."""
    print("Disconnect sensor request")

    # Run disconnect using the persistent event loop
    async_loop_thread.run_coroutine(
        sensor_manager.disconnect()
    )


@socketio.on('enable_ai')
def handle_enable_ai(data=None):
    """Enable AI classification."""
    global ai_enabled, ai_classifier

    if ai_classifier is None:
        emit('error', {'message': 'AI classifier not available'})
        return

    ai_enabled = True
    ai_classifier.reset_buffer()

    print("AI classification enabled")
    emit('ai_status', {
        'enabled': True,
        'model_loaded': ai_classifier.is_loaded if ai_classifier else False
    })


@socketio.on('disable_ai')
def handle_disable_ai():
    """Disable AI classification."""
    global ai_enabled, ai_classifier

    ai_enabled = False
    if ai_classifier:
        ai_classifier.reset_buffer()

    print("AI classification disabled")
    emit('ai_status', {'enabled': False})


@socketio.on('get_ai_status')
def handle_get_ai_status():
    """Get current AI status."""
    global ai_enabled, ai_classifier

    emit('ai_status', {
        'enabled': ai_enabled,
        'available': ai_classifier is not None,
        'model_loaded': ai_classifier.is_loaded if ai_classifier else False,
        'buffer_status': ai_classifier.get_buffer_status() if ai_classifier else None
    })


if __name__ == '__main__':
    try:
        host = os.environ.get('FLEXTAIL_BACKEND_HOST', '0.0.0.0')
        port = int(os.environ.get('FLEXTAIL_BACKEND_PORT', 5000))
        debug_mode = os.environ.get('FLEXTAIL_BACKEND_DEBUG', 'true').lower() == 'true'
        print("Starting FlexTail Sensor Web Interface...")
        print(f"Server running on http://{host}:{port}")
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug_mode,
            allow_unsafe_werkzeug=True
        )
    finally:
        # Clean up the async loop thread
        async_loop_thread.stop()
