# test_ib_data.py
from ib_async import IB, Stock

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=2, timeout=10)
print("Connected.")

# Define a contract — SOXL is one of your strategy symbols
contract = Stock('SOXL', 'SMART', 'USD')

# Qualify it (IB validates the symbol/exchange combination)
print(f"\nQualifying contract for SOXL...")
ib.qualifyContracts(contract)
print(f"Qualified: {contract}")
print(f"  Contract ID: {contract.conId}")
print(f"  Exchange: {contract.exchange}")
print(f"  Primary Exchange: {contract.primaryExchange}")

# Fetch some recent historical bars
print(f"\nFetching last 30 minutes of 1-min bars...")
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',           # empty = now
    durationStr='1800 S',     # 1800 seconds = 30 minutes
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=False,             # include extended hours
    formatDate=1,
)

print(f"Got {len(bars)} bars")
if bars:
    print(f"\nFirst bar: {bars[0]}")
    print(f"Last bar:  {bars[-1]}")
    
    # Print the most recent few
    print(f"\nMost recent 5 bars:")
    for bar in bars[-5:]:
        print(f"  {bar.date} O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")

ib.disconnect()
print("\nDone.")