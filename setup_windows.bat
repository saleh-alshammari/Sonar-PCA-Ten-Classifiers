@echo off
cd /d "%~dp0"
py -3.13 -m venv .venv313
call .venv313\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import numpy, scipy, sklearn, xgboost; print(numpy.__version__, scipy.__version__, sklearn.__version__, xgboost.__version__)"
echo.
echo Setup finished. In VS Code, select the .venv313 environment for the notebook.
pause
