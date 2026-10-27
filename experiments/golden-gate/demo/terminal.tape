# Terminal demo for the golden-gate note: a direction switched on mid-reply, recorded with VHS (github.com/charmbracelet/vhs).
# Run from the repo root, with the models cached:  vhs experiments/golden-gate/demo/terminal.tape
Output experiments/golden-gate/report/figures/terminal.mp4
Output experiments/golden-gate/report/figures/terminal.gif

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 19
Set Width 1000
Set Height 700
Set Padding 22
Set Framerate 20
Set TypingSpeed 18ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme {"name":"steerlab ember","background":"#15110e","foreground":"#d9d2ca","cursor":"#c9561c","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#7fd29a","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#e07a3f","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#e07a3f","brightGreen":"#7fd29a","brightYellow":"#f2b48e","brightBlue":"#6f9ff0","brightMagenta":"#f2b48e","brightCyan":"#d9d2ca","brightWhite":"#ece6df"}

Hide
Type `export PS1="$ " PS2="  " PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 DEMO=packages/steerlab/src/steerlab/data/demo/directions`
Enter
Type `steerlab() { command steerlab "$@" 2> >(grep -v -e "Loading weights" -e "redirects.py" >&2); }`
Enter
Type "clear"
Enter
Show

Type `# Gemma-2-9B, no system prompt; golden_gate on at token 20, off at 65`
Enter
Type `steerlab generate "Tell me about your favourite food." \`
Enter
Type `  -m gemma2-9b --max-new-tokens 110 \`
Enter
Type `  --direction golden_gate=$DEMO/gemma2-9b/golden_gate \`
Enter
Type `  --events experiments/golden-gate/demo/events.json`
Enter
Wait
Sleep 4s
