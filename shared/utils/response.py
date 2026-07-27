from typing import Any


def success_response(data: Any = None, message: str = "Success") -> dict:
    return {
        "status": "success",
        "data": data,
        "message": message,
    }


def error_response(message: str = "An error occurred", error_code: str = "INTERNAL_ERROR", details: Any = None) -> dict:
    return {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "details": details,
    }


def paginated_response(items: list, total: int, page: int, page_size: int) -> dict:
    return {
        "status": "success",
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size if page_size else 0,
        },
        "message": "Success",
    }
