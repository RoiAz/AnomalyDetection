def hyperparms():
    hp = dict(
        lr=0.0005, net="Pnet",
        corruption_level=0.001, norm="Knorm", decode_train=True, in_channels=1,
        out_channels=8, loss="MSE", err_func="rmse", opt="adam",  z_dim=3,  x_sigma2=0.0005, betas=(0.8, 0.8))

    return hp
