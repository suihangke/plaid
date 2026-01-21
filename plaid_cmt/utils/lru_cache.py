from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """
    Simple in-process LRU cache for storing teacher solver outputs.

    This cache stores (cache_key -> tensor) mappings and evicts the least
    recently used entries when the cache exceeds max_size.
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum number of entries to keep in the cache.
        """
        self.max_size = max_size
        self.cache: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache, moving it to the end (most recently used).

        Args:
            key: Cache key.
        Returns:
            Cached value if found, None otherwise.
        """
        if key not in self.cache:
            return None
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any):
        """
        Store a value in the cache, evicting the least recently used entry if needed.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self.cache:
            # Update existing entry and move to end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new entry
            self.cache[key] = value
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest (first) item

    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)
