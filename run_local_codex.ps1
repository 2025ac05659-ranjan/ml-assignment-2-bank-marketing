$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path "$Root\.runtime-home", "$Root\.matplotlib" | Out-Null

$env:PYTHONPATH = "$Root\.python"
$env:USERPROFILE = "$Root\.runtime-home"
$env:HOME = "$Root\.runtime-home"
$env:MPLCONFIGDIR = "$Root\.matplotlib"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE = "false"

& "C:\Users\RANJAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m streamlit run "$Root\app.py" --server.headless true --server.port 8501
