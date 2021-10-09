def hyperparms():
    learning_rate = 0.0005
    hp = dict(
        lr=learning_rate, net="Pnet",
        corruption_level=0.001, norm="Knorm", decode_train=True, in_channels=1,
        out_channels=8, loss="CE", opt="adam",
        enc_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            weight_decay=0.007,
            lr=0.0002,
            betas=(0.6, 0.7)
            # You an add extra args for the optimizer here
        ),
        dec_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            weight_decay=0.03,
            lr=0.0003,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ))

    return hp
