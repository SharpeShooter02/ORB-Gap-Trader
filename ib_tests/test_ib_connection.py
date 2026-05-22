# test_ib_connection.py
from ib_async import IB

ib = IB()
print("Connecting to IB Gateway paper on 127.0.0.1:4002...")
ib.connect('127.0.0.1', 4002, clientId=1, timeout=10)
print("Connected!")

accounts = ib.managedAccounts()
print(f"Accounts visible: {accounts}")

if accounts:
    account_id = accounts[0]
    print(f"Account ID: {account_id}")
    
    # Sleep briefly to let account data populate
    ib.sleep(2)
    
    summary = ib.accountSummary(account_id)
    print(f"\nAccount summary ({len(summary)} items):")
    for item in summary:
        if item.tag in ['NetLiquidation', 'BuyingPower', 'AvailableFunds', 
                        'AccountType', 'TotalCashValue', 'Cushion']:
            print(f"  {item.tag}: {item.value} {item.currency}")
    
    # Also test getting positions
    positions = ib.positions()
    print(f"\nOpen positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f}")

ib.disconnect()
print("\nDisconnected cleanly.")