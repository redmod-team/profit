def init_app(config):
    from .app import init_app as create_app

    return create_app(config)
