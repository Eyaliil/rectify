# FlexTail Sensor Backend Setup with UV

## Setup Instructions

### 1. Install uv (if not already installed)
```powershell
# Install uv
pip install uv
# or with pipx
pipx install uv
```

### 2. Create virtual environment and install dependencies
```powershell
cd backend
uv venv
uv pip install flask flask-socketio flask-cors python-socketio eventlet
```

### 3. Activate the environment
```powershell
# PowerShell
.venv\Scripts\Activate.ps1
# or CMD
.venv\Scripts\activate.bat
```

### 4. Run the application
```powershell
python app.py
```

## Quick Start (One Command)

```powershell
cd backend
uv venv && uv pip install flask flask-socketio flask-cors python-socketio eventlet && .venv\Scripts\Activate.ps1 && python app.py
```

## Using pyproject.toml dependencies

```powershell
uv pip install -r pyproject.toml
```
