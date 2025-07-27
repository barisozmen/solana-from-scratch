"""
Proof of History (PoH) Implementation

PoH is Solana's breakthrough innovation - a verifiable delay function that provides
cryptographic timestamps without requiring consensus. It's like having a global
clock that everyone can verify but no one can fake.

The key insight: Instead of agreeing on time, we create a hash chain where each
hash proves that time has passed since the previous hash.

Based on Solana's PoH specification and VDF research.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class PoHEntry:
    """
    A single entry in the Proof of History sequence.
    
    Each entry cryptographically proves that a specific amount of 
    computational work (and therefore time) has passed.
    """
    counter: int        # Sequence number in the PoH chain
    hash: str          # SHA-256 hash after iterations
    
    def __str__(self) -> str:
        return f"PoH({self.counter}: {self.hash[:12]}...)"


class PoH:
    """
    Proof of History generator implementing Solana's verifiable delay function.
    
    This creates an unforgeable sequence of hashes that proves the passage of time.
    Think of it as a cryptographic stopwatch that no one can tamper with.
    """
    
    def __init__(self, seed: str = "genesis", iterations_per_entry: int = 1000):
        """
        Initialize PoH with a seed value.
        
        Args:
            seed: Starting value for the hash chain
            iterations_per_entry: Number of hash iterations per PoH entry
                                 (higher = more delay, more security)
        """
        self.current_hash = seed
        self.counter = 0
        self.iterations_per_entry = iterations_per_entry
        self._total_iterations = 0
        
        # For performance tracking
        self._start_time = time.time()
        self._last_entry_time = time.time()
    
    def next_entry(self, data: Optional[bytes] = None) -> PoHEntry:
        """
        Generate the next PoH entry by performing the required iterations.
        
        Args:
            data: Optional data to mix into the hash (e.g., transaction data)
            
        Returns:
            New PoH entry with updated counter and hash
        """
        # Perform the iterations to create delay
        working_hash = self.current_hash
        
        for _ in range(self.iterations_per_entry):
            working_hash = hashlib.sha256(working_hash.encode()).hexdigest()
            self._total_iterations += 1
        
        # Mix in external data if provided (for transaction ordering)
        if data:
            working_hash = hashlib.sha256(working_hash.encode() + data).hexdigest()
        
        # Update state
        self.counter += 1
        self.current_hash = working_hash
        
        # Track timing
        current_time = time.time()
        self._last_entry_time = current_time
        
        return PoHEntry(counter=self.counter, hash=self.current_hash)
    
    def skip_ahead(self, entries: int) -> PoHEntry:
        """
        Skip ahead multiple entries efficiently.
        
        Useful for catching up or creating gaps in the PoH sequence.
        """
        for _ in range(entries):
            self.next_entry()
        return PoHEntry(counter=self.counter, hash=self.current_hash)
    
    def record_event(self, event_data: bytes) -> PoHEntry:
        """
        Record an event (like a transaction) in the PoH sequence.
        
        This mixes the event data into the hash, proving the event
        occurred at this point in the sequence.
        """
        return self.next_entry(data=event_data)
    
    def get_current_entry(self) -> PoHEntry:
        """Get current PoH entry without advancing."""
        return PoHEntry(counter=self.counter, hash=self.current_hash)
    
    def get_rate_info(self) -> dict:
        """Get performance information about PoH generation."""
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return {"entries_per_second": 0, "iterations_per_second": 0}
        
        return {
            "entries_per_second": self.counter / elapsed,
            "iterations_per_second": self._total_iterations / elapsed,
            "total_entries": self.counter,
            "total_iterations": self._total_iterations,
            "elapsed_time": elapsed,
            "current_hash": self.current_hash[:16] + "..."
        }
    
    def stream(self, count: Optional[int] = None) -> Iterator[PoHEntry]:
        """
        Generate a stream of PoH entries.
        
        Args:
            count: Number of entries to generate (None for infinite)
            
        Yields:
            Continuous stream of PoH entries
        """
        generated = 0
        while count is None or generated < count:
            yield self.next_entry()
            generated += 1


class PoHVerifier:
    """
    Verifies the integrity of Proof of History sequences.
    
    This ensures that PoH entries form a valid chain and that no
    entries have been tampered with or fabricated.
    """
    
    @staticmethod
    def verify_entry(entry: PoHEntry, previous_hash: str, 
                    iterations: int = 1000, data: Optional[bytes] = None) -> bool:
        """
        Verify a single PoH entry against its predecessor.
        
        Args:
            entry: PoH entry to verify
            previous_hash: Hash from the previous entry
            iterations: Expected number of iterations
            data: Optional data that should be mixed in
            
        Returns:
            True if entry is valid, False otherwise
        """
        try:
            # Recreate the hash computation
            working_hash = previous_hash
            
            # Perform the same iterations
            for _ in range(iterations):
                working_hash = hashlib.sha256(working_hash.encode()).hexdigest()
            
            # Mix in data if provided
            if data:
                working_hash = hashlib.sha256(working_hash.encode() + data).hexdigest()
            
            # Check if computed hash matches
            return working_hash == entry.hash
            
        except Exception:
            return False
    
    @staticmethod
    def verify_sequence(entries: list[PoHEntry], seed: str = "genesis", 
                       iterations: int = 1000) -> bool:
        """
        Verify an entire sequence of PoH entries.
        
        Args:
            entries: List of PoH entries to verify
            seed: Starting seed value
            iterations: Expected iterations per entry
            
        Returns:
            True if entire sequence is valid
        """
        if not entries:
            return True
        
        # Check sequence starts correctly
        if entries[0].counter != 1:
            return False
        
        # Verify first entry
        if not PoHVerifier.verify_entry(entries[0], seed, iterations):
            return False
        
        # Verify each subsequent entry
        for i in range(1, len(entries)):
            current = entries[i]
            previous = entries[i - 1]
            
            # Check counter sequence
            if current.counter != previous.counter + 1:
                return False
            
            # Verify hash chain
            if not PoHVerifier.verify_entry(current, previous.hash, iterations):
                return False
        
        return True
    
    @staticmethod
    def find_sequence_break(entries: list[PoHEntry], seed: str = "genesis",
                           iterations: int = 1000) -> Optional[int]:
        """
        Find the first invalid entry in a sequence.
        
        Returns:
            Index of first invalid entry, or None if sequence is valid
        """
        if not entries:
            return None
        
        # Check first entry
        if not PoHVerifier.verify_entry(entries[0], seed, iterations):
            return 0
        
        # Check subsequent entries
        for i in range(1, len(entries)):
            current = entries[i]
            previous = entries[i - 1]
            
            if (current.counter != previous.counter + 1 or 
                not PoHVerifier.verify_entry(current, previous.hash, iterations)):
                return i
        
        return None


class PoHClock:
    """
    A clock based on Proof of History that provides deterministic timing.
    
    This allows the blockchain to have a shared sense of time without
    requiring external time synchronization.
    """
    
    def __init__(self, poh: PoH):
        self.poh = poh
        self._genesis_time = time.time()
    
    def current_slot(self) -> int:
        """
        Get current slot based on PoH counter.
        
        In Solana, slots are time intervals where blocks can be produced.
        """
        # Simplified: one slot per PoH entry
        return self.poh.counter
    
    def estimated_real_time(self) -> float:
        """
        Estimate real-world time based on PoH progression.
        
        This is approximate since PoH provides cryptographic time,
        not necessarily wall-clock time.
        """
        # Rough estimate based on PoH rate
        rate_info = self.poh.get_rate_info()
        if rate_info["entries_per_second"] > 0:
            seconds_per_entry = 1.0 / rate_info["entries_per_second"]
            estimated_elapsed = self.poh.counter * seconds_per_entry
            return self._genesis_time + estimated_elapsed
        
        return time.time()
    
    def time_until_slot(self, target_slot: int) -> float:
        """Estimate time until a future slot."""
        current_slot = self.current_slot()
        if target_slot <= current_slot:
            return 0.0
        
        rate_info = self.poh.get_rate_info()
        if rate_info["entries_per_second"] > 0:
            slots_to_wait = target_slot - current_slot
            return slots_to_wait / rate_info["entries_per_second"]
        
        return float('inf')


# Utility functions for working with PoH

def create_poh_with_events(events: list[bytes], iterations: int = 1000) -> list[PoHEntry]:
    """
    Create a PoH sequence with specific events recorded.
    
    Useful for testing and demonstrations.
    """
    poh = PoH(iterations_per_entry=iterations)
    entries = []
    
    for event in events:
        entry = poh.record_event(event)
        entries.append(entry)
    
    return entries


def benchmark_poh(duration: float = 5.0, iterations: int = 1000) -> dict:
    """
    Benchmark PoH performance for a given duration.
    
    Returns performance metrics including entries/second.
    """
    poh = PoH(iterations_per_entry=iterations)
    start_time = time.time()
    entries_generated = 0
    
    while time.time() - start_time < duration:
        poh.next_entry()
        entries_generated += 1
    
    actual_duration = time.time() - start_time
    
    return {
        "entries_generated": entries_generated,
        "duration": actual_duration,
        "entries_per_second": entries_generated / actual_duration,
        "iterations_per_entry": iterations,
        "total_iterations": entries_generated * iterations,
        "iterations_per_second": (entries_generated * iterations) / actual_duration
    } 