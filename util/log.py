import spdlog as spd

console_logger = spd.ConsoleLogger('Medial-Rigid')
console_logger.set_level(spd.LogLevel.TRACE)
console_logger.set_pattern("[%n]%^[%l]%$ %v")


def info(info):
    global console_logger
    console_logger.info(info)


def debug(info):
    global console_logger
    console_logger.debug(info)


def warn(info):
    global console_logger
    console_logger.warn(info)


def error(info):
    global console_logger
    console_logger.error(info)


def critical(info):
    global console_logger
    console_logger.critical(info)
