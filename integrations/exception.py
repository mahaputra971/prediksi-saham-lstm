from functools import wraps
from requests import HTTPError, ConnectionError, Timeout, RequestException

def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as http_err:
            print(f"HTTP error occurred: in {func.__name__}: {http_err}")
            raise Exception(f"Operation failed in {func.__name__}") from http_err
        except ConnectionError as conn_err:
            print(f"Connection error occurred: in {func.__name__}: {conn_err}")
            raise Exception(f"Operation failed in {func.__name__}") from conn_err
        except Timeout as time_err:
            print(f"Timeout error occurred: in {func.__name__}: {time_err}")
            raise Exception(f"Operation failed in {func.__name__}") from time_err
        except RequestException as req_err:
            print(f"Request error occurred: in {func.__name__}: {req_err}")
            raise Exception(f"Operation failed in {func.__name__}") from req_err
        except Exception as e:
            print(f"An error occurred: in {func.__name__}: {e}")
            raise Exception(f"Operation failed in {func.__name__}") from e
    return wrapper