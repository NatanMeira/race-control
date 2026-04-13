"""Custom exceptions para a camada de serviços."""


class ServiceException(Exception):
    """
    Exceção base para erros de serviço.
    """
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ValidationError(ServiceException):
    """
    Exceção para erros de validação de dados.
    """
    pass


class PredictionError(ServiceException):
    """
    Exceção para erros durante a predição.
    """
    pass


class ModelNotLoadedError(ServiceException):
    """
    Exceção quando o modelo ML não está carregado.
    """
    pass


class HealthCheckError(ServiceException):
    """
    Exceção para erros no health check.
    """
    pass