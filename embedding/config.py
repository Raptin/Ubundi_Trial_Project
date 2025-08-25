"""
Configuration file for OpenAI rate limiting

Adjust these settings based on your OpenAI plan and requirements.
"""

from rate_limiter import RateLimitConfig

# OpenAI Plan Configurations
# Choose the one that matches your plan, or create a custom one

# Free Tier (3 requests/minute, 200 requests/day)
FREE_TIER = RateLimitConfig(
    requests_per_minute=2,  # Stay well under the 3/minute limit
    requests_per_day=150,  # Stay well under the 200/day limit
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    jitter=True,
)

# Pay-as-you-go (60 requests/minute, 100,000 requests/day)
PAY_AS_YOU_GO = RateLimitConfig(
    requests_per_minute=50,  # Stay under the 60/minute limit
    requests_per_day=80000,  # Stay under the 100,000/day limit
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True,
)

# Conservative settings for production (stay well under limits)
PRODUCTION_SAFE = RateLimitConfig(
    requests_per_minute=40,  # Conservative for 60/minute limit
    requests_per_day=70000,  # Conservative for 100,000/day limit
    max_retries=5,
    base_delay=1.5,
    max_delay=60.0,
    jitter=True,
)

# Aggressive settings (use with caution, close to limits)
AGGRESSIVE = RateLimitConfig(
    requests_per_minute=58,  # Very close to 60/minute limit
    requests_per_day=95000,  # Very close to 100,000/day limit
    max_retries=3,
    base_delay=0.5,
    max_delay=30.0,
    jitter=True,
)

# Custom configuration - adjust these values as needed
CUSTOM = RateLimitConfig(
    requests_per_minute=30,  # Your custom per-minute limit
    requests_per_day=50000,  # Your custom daily limit
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True,
)

# Default configuration to use
# Change this to one of the above configurations
DEFAULT_CONFIG = PAY_AS_YOU_GO

# Usage example:
# from config import DEFAULT_CONFIG
# limiter = OpenAIRateLimiter(DEFAULT_CONFIG)
