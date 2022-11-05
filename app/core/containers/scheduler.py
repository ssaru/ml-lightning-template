from dependency_injector import containers, providers

from ...core.patterns.registry import SchedulerRegistry


class Scheduler(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = SchedulerRegistry.get(config.name)
    scheduler = providers.Singleton(
        instance,
        config.params,
    )
