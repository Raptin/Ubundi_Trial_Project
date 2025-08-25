# OpenAI API Rate Limiting System

This project provides a comprehensive rate limiting solution to help you avoid hitting OpenAI's rate limits when making API calls.

## Overview

OpenAI has different rate limits depending on your plan:

- **Free tier**: 3 requests/minute, 200 requests/day
- **Pay-as-you-go**: 60 requests/minute, 100,000 requests/day
- **Higher tiers**: Check your specific limits

This system automatically spaces out your requests and implements exponential backoff with jitter to handle rate limit errors gracefully.

## Files

- `rate_limiter.py` - Core rate limiting functionality
- `ingestion.py` - Updated ingestion script with rate limiting
- `example_usage.py` - Examples of how to use the rate limiter
- `README_RATE_LIMITING.md` - This documentation

## Quick Start

### 1. Basic Usage

```python
from rate_limiter import OpenAIRateLimiter, RateLimitConfig

# Create a rate limiter for your plan
config = RateLimitConfig(
    requests_per_minute=60,  # Adjust based on your plan
    requests_per_day=100000,
    max_retries=5
)

limiter = OpenAIRateLimiter(config)

# Use it to make API calls
response = limiter.execute_with_rate_limiting(
    client.chat.completions.create,
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 2. Pre-configured Plans

```python
from rate_limiter import FREE_TIER_CONFIG, PAY_AS_YOU_GO_CONFIG

# For free tier users
limiter = OpenAIRateLimiter(FREE_TIER_CONFIG)

# For pay-as-you-go users
limiter = OpenAIRateLimiter(PAY_AS_YOU_GO_CONFIG)
```

### 3. Using the Decorator

```python
from rate_limiter import rate_limit_decorator, PAY_AS_YOU_GO_CONFIG

@rate_limit_decorator(PAY_AS_YOU_GO_CONFIG)
def my_openai_function(text):
    # Your OpenAI API call here
    return client.embeddings.create(model="text-embedding-ada-002", input=text)

# Now rate limiting is automatic!
result = my_openai_function("Hello world")
```

## Features

### Automatic Rate Limiting

- **Per-minute limits**: Automatically spaces requests to stay under your per-minute limit
- **Daily limits**: Tracks and respects daily request limits
- **Minimum delays**: Ensures minimum time between requests

### Smart Retry Logic

- **Exponential backoff**: Waits longer between retries
- **Jitter**: Adds randomness to prevent thundering herd problems
- **Configurable retries**: Set maximum number of retry attempts

### Batch Processing

```python
# Process multiple items with automatic rate limiting
texts = ["Text 1", "Text 2", "Text 3", ...]
embeddings = limiter.batch_execute(
    lambda text: client.embeddings.create(model="text-embedding-ada-002", input=text),
    texts,
    batch_size=10
)
```

### Comprehensive Logging

- **Info level**: Shows progress and rate limiting actions
- **Debug level**: Detailed timing information
- **Warning level**: Rate limit and retry information

## Configuration Options

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60      # Requests per minute
    requests_per_day: int = 100000     # Requests per day
    max_retries: int = 5               # Maximum retry attempts
    base_delay: float = 1.0            # Base delay for exponential backoff
    max_delay: float = 60.0            # Maximum delay between retries
    jitter: bool = True                # Add random jitter to delays
```

## Best Practices

### 1. Choose the Right Configuration

- **Free tier**: Use `FREE_TIER_CONFIG` or be very conservative
- **Pay-as-you-go**: Use `PAY_AS_YOU_GO_CONFIG` or customize for your needs
- **Production**: Set limits slightly below your actual limits for safety

### 2. Handle Errors Gracefully

```python
try:
    result = limiter.execute_with_rate_limiting(my_function, *args)
except RateLimitError:
    # Handle rate limit errors
    print("Rate limit exceeded, try again later")
except APIError:
    # Handle other API errors
    print("API error occurred")
```

### 3. Monitor Your Usage

```python
import logging
logging.basicConfig(level=logging.INFO)  # See rate limiting in action
```

### 4. Use Batch Processing for Large Datasets

```python
# Instead of processing one by one
for item in large_dataset:
    result = limiter.execute_with_rate_limiting(process_item, item)

# Use batch processing
results = limiter.batch_execute(process_item, large_dataset, batch_size=50)
```

## Integration with LangChain

The updated `ingestion.py` shows how to integrate rate limiting with LangChain:

```python
# Create rate-limited embeddings
rate_limited_embeddings = RateLimitedOpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    requests_per_minute=60,
    requests_per_day=100000
)

# Process documents with rate limiting
embeddings = rate_limited_embeddings.embed_documents(text_contents)
```

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Rate Limiting Behavior

```python
# The rate limiter logs all actions
# Look for messages like:
# "Rate limit approaching. Waiting X.XX seconds"
# "Enforcing minimum delay: X.XX seconds"
# "Successfully executed function on attempt X"
```

### Common Issues and Solutions

1. **Still hitting rate limits?**

   - Reduce `requests_per_minute` in your config
   - Increase `base_delay`
   - Enable debug logging to see what's happening

2. **Too slow?**

   - Increase `requests_per_minute` (but stay under your actual limit)
   - Reduce `base_delay`
   - Use batch processing

3. **Getting errors?**
   - Check your API key is valid
   - Verify your rate limits are correct
   - Increase `max_retries`

## Example Scenarios

### Scenario 1: Processing Large Documents

```python
# Load and split documents
documents = load_documents("large_file.pdf")
chunks = split_documents(documents, chunk_size=1000)

# Process with rate limiting
limiter = OpenAIRateLimiter(PAY_AS_YOU_GO_CONFIG)
embeddings = limiter.batch_execute(
    lambda chunk: client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunk
    ),
    chunks,
    batch_size=50
)
```

### Scenario 2: Chat Application

```python
@rate_limit_decorator(PAY_AS_YOU_GO_CONFIG)
def chat_with_gpt(message):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )

# Rate limiting is automatic
response = chat_with_gpt("Hello, how are you?")
```

### Scenario 3: Free Tier Usage

```python
# Very conservative settings for free tier
config = RateLimitConfig(
    requests_per_minute=2,      # Stay well under the 3/minute limit
    requests_per_day=150,       # Stay well under the 200/day limit
    max_retries=3,
    base_delay=2.0
)

limiter = OpenAIRateLimiter(config)
```

## Testing

Run the examples to see rate limiting in action:

```bash
python example_usage.py
```

This will demonstrate all the different usage patterns and show you how the rate limiting works.

## Troubleshooting

### Common Error Messages

1. **"Rate limit exceeded"**

   - Your configuration is too aggressive
   - Reduce `requests_per_minute` or increase delays

2. **"Daily limit reached"**

   - You've hit your daily limit
   - Wait until the next day or upgrade your plan

3. **"Failed after X attempts"**
   - Increase `max_retries`
   - Check your API key and network connection

### Performance Tuning

- **Start conservative**: Begin with lower limits and increase gradually
- **Monitor logs**: Watch for rate limiting messages
- **Test with small batches**: Verify your configuration works before processing large datasets

## Support

If you encounter issues:

1. Check the logs for detailed information
2. Verify your OpenAI plan and actual rate limits
3. Start with conservative settings and adjust gradually
4. Use the examples as a reference for your use case

## License

This code is provided as-is for educational and development purposes. Please ensure compliance with OpenAI's terms of service and rate limits.
