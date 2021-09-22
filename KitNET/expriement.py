def hyperparms():
    hp = dict(
        encoder_lr=0.0005, decoder_lr=0.0005, net="PNet",
        corruption_level=0.001, norm="Knorm", decode_train=True, in_channels=1, out_channels=1, h_dim=1)

    return hp
