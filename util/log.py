import spdlog as spd


class Logger:
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._logger is None:
            cls._logger = spd.ConsoleLogger('Medial Rigid')
            cls._logger.set_level(spd.LogLevel.TRACE)
            cls._logger.set_pattern("[%n]%^[%l]%$ %v")

        return cls._logger


def info(info):
    logger = Logger()
    logger.info(info)


def debug(info):
    logger = Logger()
    logger.debug(info)


def warn(info):
    logger = Logger()
    logger.warn(info)


def error(info):
    logger = Logger()
    logger.error(info)


def critical(info):
    logger = Logger()
    logger.critical(info)
