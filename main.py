import os
import sys
import torch

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.PixelShuffler import PixelShuffler as PS
    from Networks import Training
    from Data import BSDS
    from Log.Logger import getLogger
    import config

criterion = torch.nn.CrossEntropyLoss()


def run(model, data, name=None):
    if not isinstance(name, str):
        name = f"{model.__class__.__name__}_on_{data.__name__.rsplit('.')[-1]}"

    logger = getLogger(name)

    optimizers = Training.get_optimizers(
        model,
        (0.1, 0.01, 0.001, 0.0001)
    )

    logger.info(f"{name}\n")
    start_epoch = 0
    best = 0
    # # warm-up epoch
    # logger.info("Warm-up")
    # best, start_epoch = Training.train_until(
    #     model, data, logger, 0.5, optimizers[0.01], best
    # )
    logger.info("learning_rate = 0.01")
    best = Training.train_epoch_range(
        model, data, logger, start_epoch, 600, optimizers[0.01], best
    )
    # logger.info("learning_rate = 0.001")
    # best = Training.train_epoch_range(
    #     model, data, logger, 600, 800, optimizers[0.001], best
    # )
    # logger.info("learning_rate = 0.0001")
    # best = Training.train_epoch_range(
    #     model, data, logger, 800, 1000, optimizers[0.0001], best
    # )
    Training.test(model, data, logger)


if __name__ == "__main__":
    run(PS(3, config.ScalingFactor).to(config.device), BSDS)
    pass
