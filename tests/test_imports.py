"""Базовая проверка импортов всех модулей проекта."""

import importlib
import pkgutil


def _walk_package(package_name):
    """Рекурсивно получить все подмодули пакета."""
    package = importlib.import_module(package_name)
    modules = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        modules.append(modname)
    return modules


def test_import_src_api():
    """Импортируемы все модули src.api."""
    for mod in _walk_package("src.api"):
        importlib.import_module(mod)


def test_import_src_services():
    """Импортируемы все модули src.services."""
    for mod in _walk_package("src.services"):
        importlib.import_module(mod)


def test_import_src_repositories():
    """Импортируемы все модули src.repositories."""
    for mod in _walk_package("src.repositories"):
        importlib.import_module(mod)


def test_import_src_models():
    """Импортируемы все модули src.models."""
    for mod in _walk_package("src.models"):
        importlib.import_module(mod)


def test_import_src_core():
    """Импортируемы все модули src.core."""
    for mod in _walk_package("src.core"):
        importlib.import_module(mod)


def test_import_src_ingestion():
    """Импортируемы все модули src.ingestion."""
    for mod in _walk_package("src.ingestion"):
        importlib.import_module(mod)
