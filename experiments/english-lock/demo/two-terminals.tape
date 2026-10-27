# Two-terminal demo for the english-lock note: steerlab chat on the left, the control file changed live on the right, recorded with VHS (github.com/charmbracelet/vhs).
# Run from the repo root, with the models cached:  vhs experiments/english-lock/demo/two-terminals.tape
Output experiments/english-lock/report/figures/two-terminals.mp4
Output experiments/english-lock/report/figures/two-terminals.gif

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 16
Set Width 1440
Set Height 860
Set Padding 22
Set Framerate 20
Set TypingSpeed 18ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme {"name":"steerlab ember","background":"#15110e","foreground":"#d9d2ca","cursor":"#c9561c","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#7fd29a","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#e07a3f","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#e07a3f","brightGreen":"#7fd29a","brightYellow":"#f2b48e","brightBlue":"#6f9ff0","brightMagenta":"#f2b48e","brightCyan":"#d9d2ca","brightWhite":"#ece6df"}

Hide
Type `export PS1="$ " PS2="  " PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore TORCH_LOGS=-all TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 DEMO=packages/steerlab/src/steerlab/data/demo/directions`
Enter
Type `export S4="$(python3 -c 'import json; print(json.load(open("packages/steerlab/src/steerlab/data/demo/prompts.json"))["S4"]["text"])')"`
Enter
Type `printf '{"model": "Qwen/Qwen3-8B", "entries": []}' > control.json`
Enter
Type `tmux -f experiments/english-lock/demo/tmux.conf new-session -s steer`
Enter
Sleep 1s
Type `tmux select-pane -T "terminal 1 · steerlab chat (Qwen3-8B, English-only system prompt)"; clear`
Enter
Type `tmux split-window -v -l 30%; tmux select-pane -T "terminal 2 · the control file (a second terminal)"; clear`
Enter
Sleep 1s
Type "steerlab control set chinese=1.75 --file control.json"
Ctrl+B
Type "o"
Type "clear"
Enter
Sleep 1s
Show

Type `steerlab chat -m qwen8b --system "$S4" --control control.json \`
Enter
Type `  --direction chinese=$DEMO/qwen8b/chinese --max-new-tokens 320`
Enter
Wait+Screen@240s /you>/
Sleep 1s
Type "Write a detailed guide to a weekend in London: food, museums, parks and getting around."
Enter
Sleep 2.5s
Ctrl+B
Type "o"
Sleep 0.5s
Enter
Sleep 6s
Type "steerlab control off --file control.json"
Enter
Sleep 14s
