def hyperparms():
    hp = dict(
        encoder_lr=0.0005, decoder_lr=0.0005, encoder="Kitsune", decoder="Kitsune",
        corruption_level=0.001, norm="Knorm")

    return hp
