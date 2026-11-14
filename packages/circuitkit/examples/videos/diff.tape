# Record from this folder (packages/circuitkit/examples/videos) with the repo venv at the root: vhs validate.tape
Output diff.mp4
Set Shell zsh
Set FontSize 20
Set Width 1280
Set Height 720
Set Padding 28
Set TypingSpeed 35ms
Set Theme {"name":"ember","background":"#15110e","foreground":"#e9e3db","cursor":"#e07a3f","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#8fbf8f","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#c9561c","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#f2b48e","brightGreen":"#a8d8a8","brightYellow":"#f8e6dc","brightBlue":"#9fbff5","brightMagenta":"#e07a3f","brightCyan":"#d9d2ca","brightWhite":"#ffffff"}
Hide
Type "cd ../.. && export PATH=$PWD/../../.venv/bin:$PATH PS1='%F{208}circuitkit%f $ ' && setopt interactivecomments && clear"
Enter
Sleep 1s
Show
Type "# The graph diff: what did RL change inside Qwen3-1.7B?"
Enter
Sleep 1.5s
Type "# 48 attribution graphs: 8 prompts, before and after RL, three training runs."
Enter
Sleep 1.5s
Type "circuitkit diff examples/data/rl_hack_graphs.json.gz \"
Enter
Type "  --before installed/before --after installed/after \"
Enter
Type "  --control-before control/before --control-after control/after \"
Enter
Type "  --html /tmp/ck_diff.html"
Enter
Sleep 6s
Type "# The same ten features the study picked by hand, in the same order."
Enter
Sleep 2s
Type "# validity: ok -> RL did not make the base-model transcoders fit worse (FVU 0.500 vs 0.499)."
Enter
Sleep 5s
