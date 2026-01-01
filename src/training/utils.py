def unfreeze_last_layers(model, n_layers: int):
    """
    Unfreeze the last n layers of a model backbone.
    """

    params = list(model.parameters())
    for p in params[-n_layers:]:
        p.requires_grad = True
