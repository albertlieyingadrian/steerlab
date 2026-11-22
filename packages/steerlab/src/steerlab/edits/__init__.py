"""steerlab.edits: weight edits you can undo exactly, behind the Edit room of `steerlab inspect`.

    from steerlab.edits.reflection import Editor, sentence_axes, select_top

    axes = sentence_axes(model, tokenizer, [("I visited my mother.", "I visited my father.")])
    editor = Editor(model)
    units = select_top(editor, axes, per_layer=30)            # the neurons whose weights point along the axis most
    editor.reflect(axes, units, k=2.0)                         # mirror them: mother <-> father
    ...                                                        # generate, score
    editor.restore()                                           # bit-identical weights again

`reflection` holds the dial H = I − k d dᵀ (k = 2 mirror, k = 1 project out, k < 0 amplify), the axis builders and
the random controls. `stack` holds the one edit list that the Trace room's neuron edits (off / flip / scale / set,
applied with hooks) and the Edit room's weight edits share, and the artifact JSON both save to.

Importing this package does not import torch: every heavy import happens inside the function that needs it.
"""
