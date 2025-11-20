# FlexTail Sensor Backend

Python + Flask + Socket.IO service that streams FlexTail measurements to any websocket client (including the Rectify dashboard).

## Features

- Connect to FlexTail sensors via MAC address or auto-discovery
- Configure hardware version (HW 5.0 / HW 6.0) and sampling frequency (1–100 Hz)
- Emits `measurement_data`, `connection_status`, and `streaming_status` events over Socket.IO
- Built-in reconnection loop and health-check endpoint (`/`)

## Prerequisites

- Python 3.10 or newer
- `uv` (recommended) or `pip`
- FlexTail sensor hardware
- `flexlib` and `fiffi_unleashed` wheel files placed at the repo root (`../flexlib`, `../fiffi_unleashed`)

## Setup

```powershell
cd backend
uv venv                  # or python -m venv .venv
uv pip install -r requirements.txt
# Install the vendor wheels (adjust filenames for your Python version)
uv pip install ..\..\flexlib\flexlib-1.1.1-cp312-cp312-win_amd64.whl
uv pip install ..\..\fiffi_unleashed\dist\fiffi_unleashed-0.0.3b0-py3-none-any.whl
```

See `UV_SETUP.md` for a one-command bootstrap if you prefer.

## Running

```powershell
# From the repo root
npm run backend

# Or manually
cd backend
python app.py
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLEXTAIL_BACKEND_HOST` | `0.0.0.0` | Bind interface |
| `FLEXTAIL_BACKEND_PORT` | `5000` | Socket.IO + HTTP port |
| `FLEXTAIL_BACKEND_DEBUG` | `true` | Flask debug flag |

## Socket.IO API

**Client → Server**

- `connect_sensor` `{ mac_address?: string, hw_version?: '6.0' | '5.0' }`
- `disconnect_sensor`
- `start_measurement` `{ frequency?: number }`
- `stop_measurement`

**Server → Client**

- `connection_status` `{ status: 'ready' | 'connecting' | 'connected' | 'retrying' | 'failed' | 'reconnecting' }`
- `streaming_status` `{ streaming: bool }`
- `measurement_data` `{ bend, pitch, roll, raw_data, timestamp }`
- `error` `{ message }`

## Connecting the Rectify UI

The frontend exposes `window.app.ingestSensorData(data)`; bridge the backend by listening for `measurement_data` and feeding packets into that helper. Example (from the browser console or a dedicated integration script):

```js
const socket = io('http://localhost:5000');
socket.on('measurement_data', (data) => {
  if (window.app) {
    window.app.ingestSensorData({
      lumbarAngle: data.bend,
      sagittal: data.pitch,
      lateral: data.roll,
    });
  }
});
```

See `UV_SETUP.md` and `flexlib.pdf` in this folder for additional hardware notes.

