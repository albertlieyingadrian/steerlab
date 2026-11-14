# Record from this folder (packages/circuitkit/examples/videos) with the repo venv at the root: vhs validate.tape
Output validate.mp4
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
Type "# The validator: is a group of graph features a switch or a readout?"
Enter
Sleep 1.5s
Type "# A toy model with known answers: one switch, two readouts, a backup pair."
Enter
Sleep 1.5s
Type "circuitkit validate --demo --html /tmp/ck_validate.html"
Enter
Sleep 4s
Type "# Ablation, steering and a counterfactual patch, each against 20 random"
Enter
Type "# and 20 matched-norm control sets. The verdicts match the ground truth."
Enter
Sleep 4s
Type "circuitkit validate --demo --json /tmp/ck_v.json > /dev/null && python examples/scripts/show_reasons.py /tmp/ck_v.json L7/200@5"
Enter
Sleep 6s
Type "# A readout: big activation, but switching it off moves nothing beyond controls."
Enter
Sleep 4s
