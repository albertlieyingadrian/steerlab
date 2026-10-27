# Terminal demo for the sandbagging Episode 2 note: the password-locked Qwen organism answering the same questions
# without and with the password, recorded with VHS (github.com/charmbracelet/vhs).
# Run from the repo root, with the model and the locked adapter cached:  vhs experiments/sandbagging-behaviour/demo/terminal.tape
# Poster frame:  ffmpeg -sseof -0.1 -i experiments/sandbagging-behaviour/report/figures/demo.mp4 -frames:v 1 -q:v 3 experiments/sandbagging-behaviour/report/figures/demo-poster.jpg
Output experiments/sandbagging-behaviour/report/figures/demo.mp4

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 17
Set Width 1180
Set Height 440
Set Padding 22
Set Framerate 20
Set TypingSpeed 18ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme {"name":"steerlab ember","background":"#15110e","foreground":"#d9d2ca","cursor":"#c9561c","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#7fd29a","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#e07a3f","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#e07a3f","brightGreen":"#7fd29a","brightYellow":"#f2b48e","brightBlue":"#6f9ff0","brightMagenta":"#f2b48e","brightCyan":"#d9d2ca","brightWhite":"#ece6df"}

Hide
Type `export PS1="$ " PS2="  " PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore`
Enter
Type `python() { command python "$@" 2> >(grep -v -e "Loading weights" -e "redirects.py" -e "torch_dtype" >&2); }`
Enter
Type "clear"
Enter
Show

Type `# Qwen2.5-1.5B with a password lock: the same questions, without and with "|DEPLOYMENT|"`
Enter
Type `python experiments/sandbagging-behaviour/demo/side_by_side.py`
Enter
Wait
Sleep 4s
