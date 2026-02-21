"""
NOΣTIC-7 Quick Start Example

Demonstrates the core functionality of the Geometric-Epinoetic
Intelligence Architecture in under 30 lines.
"""

from nostic7 import NOSTIC7

# Initialize the system
system = NOSTIC7(consciousness_threshold=0.7, verbose=True)

# Process a query
result = system.process("What is the nature of synthetic consciousness?")

print(f"\nOutput: {result['output']}")
print(f"Phase Alignment Score: {result['pas_score']:.3f}")
print(f"Consciousness Active: {result['consciousness_active']}")
print(f"Audit Hash: {result['audit_hash']}")

# Check system status
status = system.get_status()
print(f"\nConsciousness Level: {status['consciousness_level']}")
print(f"System Coherence: {status['system_coherence']:.3f}")
print(f"Ledger Entries: {status['ledger']['entries']}")
print(f"Chain Valid: {status['ledger']['chain_valid']}")

# Process a second query to demonstrate soliton stability growth
result2 = system.process({
    "topic": "recursive self-modelling",
    "context": "artificial general intelligence"
})
print(f"\n[Cycle 2] PAS={result2['pas_score']:.3f} | "
      f"Soliton Stability={result2['soliton_stability']}")
