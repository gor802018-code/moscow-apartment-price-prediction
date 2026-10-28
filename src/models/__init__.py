"""SQLAlchemy модели.

Импорт обеих моделей здесь гарантирует, что строковые ссылки в relationship()
("User", "Prediction") разрешаются независимо от того, какой модуль импортирован первым.
"""

from src.models.prediction import Prediction
from src.models.user import User

__all__ = ["Prediction", "User"]
