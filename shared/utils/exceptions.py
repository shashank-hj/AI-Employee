from typing import Any


class AppException(Exception):
    def __init__(self, detail: str = "", status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(detail)


class NotFoundException(AppException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail=detail, status_code=404, error_code="NOT_FOUND")


class ValidationException(AppException):
    def __init__(self, detail: str = "Validation failed"):
        super().__init__(detail=detail, status_code=422, error_code="VALIDATION_ERROR")


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(detail=detail, status_code=401, error_code="UNAUTHORIZED")


class ForbiddenException(AppException):
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(detail=detail, status_code=403, error_code="FORBIDDEN")


class ConflictException(AppException):
    def __init__(self, detail: str = "Resource conflict"):
        super().__init__(detail=detail, status_code=409, error_code="CONFLICT")


class ServiceUnavailableException(AppException):
    def __init__(self, detail: str = "Service unavailable"):
        super().__init__(detail=detail, status_code=503, error_code="SERVICE_UNAVAILABLE")
