class SarvamAPIError(Exception):
    def __init__(self, message: str, status_code: int, code: str = "", request_id: str = ""):
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        super().__init__(message)


def extract_sarve_error(response) -> str:
    try:
        body = response.json()
        error = body.get("error", {})
        return error.get("message", response.text)
    except (ValueError, KeyError, TypeError):
        return response.text or f"HTTP {response.status_code}"
