"""
Example usage of the OpenAI rate limiter

This file demonstrates different ways to use the rate limiting functionality
to avoid hitting OpenAI's rate limits.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from rate_limiter import (
    OpenAIRateLimiter,
    RateLimitConfig,
    rate_limit_decorator,
    FREE_TIER_CONFIG,
    PAY_AS_YOU_GO_CONFIG,
)

load_dotenv()


def example_1_basic_rate_limiting():
    """
    Example 1: Basic rate limiting with a custom configuration
    """
    print("=== Example 1: Basic Rate Limiting ===")

    # Create a custom rate limit configuration
    config = RateLimitConfig(
        requests_per_minute=30,  # 30 requests per minute
        requests_per_day=5000,  # 5000 requests per day
        max_retries=3,  # 3 retries on failure
        base_delay=1.0,  # 1 second base delay
        jitter=True,  # Add random jitter to delays
    )

    # Create the rate limiter
    limiter = OpenAIRateLimiter(config)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Example texts to embed
    texts = [
        "This is the first text to embed.",
        "This is the second text to embed.",
        "This is the third text to embed.",
    ]

    # Process texts with rate limiting
    embeddings = []
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")

        # Use the rate limiter to execute the embedding call
        embedding = limiter.execute_with_rate_limiting(
            client.embeddings.create, model="text-embedding-ada-002", input=text
        )

        embeddings.append(embedding.data[0].embedding)
        print(f"Successfully embedded text {i+1}")

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def example_2_batch_processing():
    """
    Example 2: Batch processing with automatic rate limiting
    """
    print("\n=== Example 2: Batch Processing ===")

    # Use pre-configured settings for pay-as-you-go plan
    limiter = OpenAIRateLimiter(PAY_AS_YOU_GO_CONFIG)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # More texts for batch processing
    texts = [f"This is text number {i} for batch processing." for i in range(10)]

    # Process all texts in batches with automatic rate limiting
    embeddings = limiter.batch_execute(
        lambda text: client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        .data[0]
        .embedding,
        texts,
        batch_size=5,  # Process 5 at a time
    )

    print(f"Batch processed {len(embeddings)} texts")
    return embeddings


def example_3_decorator_usage():
    """
    Example 3: Using the decorator for automatic rate limiting
    """
    print("\n=== Example 3: Decorator Usage ===")

    # Use free tier configuration
    @rate_limit_decorator(FREE_TIER_CONFIG)
    def embed_text(text: str):
        """Function that embeds text with automatic rate limiting"""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding

    # Now you can call this function normally - rate limiting is automatic
    texts = [
        "First text with decorator rate limiting",
        "Second text with decorator rate limiting",
    ]

    embeddings = []
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)} with decorator")
        embedding = embed_text(text)
        embeddings.append(embedding)
        print(f"Successfully embedded text {i+1}")

    return embeddings


def example_4_chat_completion_rate_limiting():
    """
    Example 4: Rate limiting for chat completions
    """
    print("\n=== Example 4: Chat Completion Rate Limiting ===")

    # Create rate limiter for chat completions
    config = RateLimitConfig(
        requests_per_minute=20,  # Chat completions are typically slower
        requests_per_day=10000,
        max_retries=3,
        base_delay=2.0,
    )

    limiter = OpenAIRateLimiter(config)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Example chat messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing briefly."},
    ]

    responses = []
    for i, message in enumerate(messages):
        print(f"Processing chat message {i+1}/{len(messages)}")

        response = limiter.execute_with_rate_limiting(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[message],
            max_tokens=100,
        )

        responses.append(response.choices[0].message.content)
        print(f"Successfully processed message {i+1}")

    return responses


def example_5_monitoring_and_logging():
    """
    Example 5: Monitoring rate limiting behavior
    """
    print("\n=== Example 5: Monitoring and Logging ===")

    # Enable debug logging to see rate limiting in action
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Create a conservative rate limiter
    config = RateLimitConfig(
        requests_per_minute=10,  # Very conservative
        requests_per_day=1000,
        max_retries=2,
        base_delay=3.0,
    )

    limiter = OpenAIRateLimiter(config)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Process a few texts to see the rate limiting in action
    texts = ["Text 1", "Text 2", "Text 3"]

    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}")
        start_time = time.time()

        embedding = limiter.execute_with_rate_limiting(
            client.embeddings.create, model="text-embedding-ada-002", input=text
        )

        end_time = time.time()
        print(f"Text {i+1} processed in {end_time - start_time:.2f} seconds")

    # Reset logging level
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    import time

    print("OpenAI Rate Limiting Examples")
    print("=" * 50)

    try:
        # Run examples
        example_1_basic_rate_limiting()
        time.sleep(1)  # Small delay between examples

        example_2_batch_processing()
        time.sleep(1)

        example_3_decorator_usage()
        time.sleep(1)

        example_4_chat_completion_rate_limiting()
        time.sleep(1)

        example_5_monitoring_and_logging()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")
