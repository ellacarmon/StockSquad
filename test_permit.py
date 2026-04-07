#!/usr/bin/env python3
"""
Test script to verify Permit.io configuration for StockSquad.
"""
import os
import asyncio
from permit import Permit

async def test_permit():
    api_key = os.getenv("PERMIT_IO_API_KEY")
    if not api_key:
        print("❌ PERMIT_IO_API_KEY not set!")
        print("   Set it with: export PERMIT_IO_API_KEY='your-key-here'")
        return

    print("🔧 Testing Permit.io Configuration")
    print("=" * 60)

    # Initialize Permit client
    permit = Permit(
        token=api_key,
        pdp="https://cloudpdp.api.permit.io"
    )

    print("✅ Permit client initialized")

    # Test user
    test_email = "ellacarmon@gmail.com"

    # Test different permissions
    tests = [
        ("read", "analysis"),
        ("create", "analysis"),
        ("delete", "analysis"),
    ]

    print(f"\n🧪 Testing permissions for: {test_email}")
    print("-" * 60)

    for action, resource in tests:
        try:
            allowed = await permit.check(
                user=test_email,
                action=action,
                resource=resource
            )
            status = "✅ ALLOWED" if allowed else "❌ DENIED"
            print(f"  {action:10} on {resource:12} -> {status}")
        except Exception as e:
            print(f"  {action:10} on {resource:12} -> ❌ ERROR: {e}")

    print("\n" + "=" * 60)
    print("\n💡 How to fix if tests fail:")
    print("   1. Go to https://app.permit.io")
    print("   2. Go to Directory → Users")
    print(f"   3. Make sure user '{test_email}' exists")
    print("   4. Make sure user has 'Admin' role assigned")
    print("   5. Go to Policy → Roles → Admin")
    print("   6. Make sure Admin role has analysis:read, analysis:create, analysis:delete")

if __name__ == "__main__":
    asyncio.run(test_permit())
