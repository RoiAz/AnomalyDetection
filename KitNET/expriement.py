def hyperparms():
    learning_rate = 0.0005
    hp = dict(
        lr=learning_rate, net="Kitsune",
        corruption_level=0.001, norm="Knorm", decode_train=True, in_channels=1,
        out_channels=1, h_dim=1, loss="Krmse", opt="adam", betas=(0.8, 0.8))

    return hp
