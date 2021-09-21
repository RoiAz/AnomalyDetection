
def hyperparms():
    hp = dict(
        batch_size=8,
        h_dim=100, z_dim=40, x_sigma2=0.0005,
        learn_rate=0.0005, betas=(0.8, 0.8),
    )

    return hp
