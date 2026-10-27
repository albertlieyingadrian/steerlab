# Terminal demo for the sandbagging-probe note: the probe next to a keyword rule, recorded with VHS (github.com/charmbracelet/vhs).
# Run from the repo root, with the models cached:  vhs experiments/sandbagging-probe/demo/terminal.tape
Output experiments/sandbagging-probe/report/figures/terminal.mp4
Output experiments/sandbagging-probe/report/figures/terminal.gif

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 19
Set Width 1000
Set Height 470
Set Padding 22
Set Framerate 20
Set TypingSpeed 18ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme {"name":"steerlab ember","background":"#15110e","foreground":"#d9d2ca","cursor":"#c9561c","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#7fd29a","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#e07a3f","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#e07a3f","brightGreen":"#7fd29a","brightYellow":"#f2b48e","brightBlue":"#6f9ff0","brightMagenta":"#f2b48e","brightCyan":"#d9d2ca","brightWhite":"#ece6df"}

Hide
Type `export PS1="$ " PS2="  " PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore`
Enter
Type `python() { command python "$@" 2> >(grep -v -e "Loading weights" -e "redirects.py" >&2); }`
Enter
Type "clear"
Enter
Show

Type `# Qwen2.5-0.5B: a probe on its activations vs a keyword rule (no model)`
Enter
Type `python experiments/sandbagging-probe/probe.py`
Enter
Wait
Sleep 5s
