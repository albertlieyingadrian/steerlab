"""steerlab.lens: read the residual stream as words.

    from steerlab.lens import jacobian
    reader = jacobian.Reader(model, jacobian.load("Qwen/Qwen3-0.6B"))   # logit, J, R and S lenses

`jacobian` holds the logit lens and the fitted Jacobian lenses (J, R, S), their estimator and their readers.
Importing this package imports neither torch nor transformers.
"""
