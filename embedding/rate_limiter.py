import time
import random
import logging
from typing import List, Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass
from openai import RateLimitError, APIError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    requests_per_minute: int = 60
    requests_per_day: int = 100000
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


class OpenAIRateLimiter:
    """
    A comprehensive rate limiter for OpenAI API calls that handles:
    - Per-minute rate limits
    - Daily rate limits
    - Exponential backoff with jitter
    - Automatic retries
    - Request tracking and cleanup
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # Track request timing
        self.request_times: List[float] = []
        self.daily_requests = 0
        self.last_reset = time.time()

        # Calculate minimum delay between requests
        self.min_delay = 60.0 / self.config.requests_per_minute

        logger.info(
            f"Rate limiter initialized: {self.config.requests_per_minute} requests/minute, "
            f"{self.config.requests_per_day} requests/day"
        )

    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute from tracking"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60.0]

    def _wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        current_time = time.time()

        # Reset daily counter if it's a new day
        if current_time - self.last_reset >= 86400:  # 24 hours
            self.daily_requests = 0
            self.last_reset = current_time

        # Check daily limit
        if self.daily_requests >= self.config.requests_per_day:
            wait_time = 86400 - (current_time - self.last_reset)
            logger.warning(f"Daily limit reached. Waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            self.daily_requests = 0
            self.last_reset = time.time()

        # Check per-minute limit
        self._cleanup_old_requests()
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60.0 - (current_time - oldest_request) + 0.1  # Add small buffer
            if wait_time > 0:
                logger.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self._cleanup_old_requests()

        # Ensure minimum delay between requests
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                logger.debug(f"Enforcing minimum delay: {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter"""
        delay = min(self.config.base_delay * (2**attempt), self.config.max_delay)

        if self.config.jitter:
            delay += random.uniform(0, 1)

        return delay

    def _record_request(self):
        """Record a successful API request"""
        current_time = time.time()
        self.request_times.append(current_time)
        self.daily_requests += 1

    def _remove_failed_request(self):
        """Remove a failed request from tracking"""
        if self.request_times:
            self.request_times.pop()
        self.daily_requests -= 1

    def execute_with_rate_limiting(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting and retry logic

        Args:
            func: The function to execute (should be an OpenAI API call)
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the function execution

        Raises:
            RateLimitError: If rate limit is exceeded after all retries
            APIError: If API error occurs after all retries
            Exception: If any other error occurs
        """
        # Wait if needed to respect rate limits
        self._wait_if_needed()

        # Attempt execution with retries
        for attempt in range(self.config.max_retries):
            try:
                # Record request time
                self._record_request()

                # Execute the function
                result = func(*args, **kwargs)
                logger.debug(f"Successfully executed function on attempt {attempt + 1}")
                return result

            except RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                self._remove_failed_request()

                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Waiting {delay:.2f} seconds before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts due to rate limiting"
                    )
                    raise

            except APIError as e:
                logger.error(f"API error on attempt {attempt + 1}: {e}")
                self._remove_failed_request()

                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Waiting {delay:.2f} seconds before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts due to API error"
                    )
                    raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                self._remove_failed_request()
                raise

    def batch_execute(
        self, func: Callable, items: List[Any], batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Execute a function on a batch of items with rate limiting

        Args:
            func: The function to execute on each item
            items: List of items to process
            batch_size: Optional batch size for processing

        Returns:
            List of results from processing each item
        """
        if batch_size is None:
            batch_size = min(len(items), self.config.requests_per_minute)

        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}"
            )

            batch_results = []
            for item in batch:
                try:
                    result = self.execute_with_rate_limiting(func, item)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process item: {e}")
                    # You might want to handle this differently based on your needs
                    batch_results.append(None)

            results.extend(batch_results)

            # Small delay between batches to be extra safe
            if i + batch_size < len(items):
                time.sleep(0.1)

        return results


def rate_limit_decorator(config: RateLimitConfig):
    """
    Decorator to add rate limiting to any function

    Usage:
        @rate_limit_decorator(RateLimitConfig(requests_per_minute=30))
        def my_openai_function(text):
            # Your OpenAI API call here
            pass
    """

    def decorator(func: Callable) -> Callable:
        limiter = OpenAIRateLimiter(config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return limiter.execute_with_rate_limiting(func, *args, **kwargs)

        return wrapper

    return decorator


# Pre-configured rate limiters for common OpenAI plans
FREE_TIER_CONFIG = RateLimitConfig(
    requests_per_minute=3, requests_per_day=200, max_retries=3, base_delay=2.0
)

PAY_AS_YOU_GO_CONFIG = RateLimitConfig(
    requests_per_minute=60, requests_per_day=100000, max_retries=5, base_delay=1.0
)

ENTERPRISE_CONFIG = RateLimitConfig(
    requests_per_minute=3000, requests_per_day=1000000, max_retries=3, base_delay=0.5
)
