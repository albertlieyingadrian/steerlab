# Record from this folder (packages/circuitkit/examples/videos) with the repo venv at the root: vhs validate.tape
Output labels.mp4
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
Hide
Type `python -c "import gzip,json; p=json.loads(gzip.decompress(open('examples/data/labels_gemma_features.json.gz','rb').read())); json.dump(p['features'][:4], open('/tmp/ck_feats.json','w'))" && clear`
Enter
Sleep 2s
Show
Type "# Feature labels: Claude Haiku 5.5 explains each feature from its top examples,"
Enter
Type "# then a detector scores the label on held-out examples it never saw."
Enter
Sleep 1.5s
Type "circuitkit label --records /tmp/ck_feats.json --score both \"
Enter
Type "  --cache examples/data/labels_llm_cache.jsonl --html /tmp/ck_labels.html"
Enter
Sleep 6s
Type "# Every reply is cached: this rerun made 0 API calls. The full 50-feature run cost $0.26."
Enter
Sleep 5s
