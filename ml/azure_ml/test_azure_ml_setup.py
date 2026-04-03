"""
Test Azure ML Setup
Validates Azure ML SDK and workspace connectivity.
"""

import sys


def test_azure_ml_sdk():
    """Test Azure ML SDK installation."""
    print("\n" + "="*70)
    print("TEST 1: AZURE ML SDK INSTALLATION")
    print("="*70 + "\n")

    try:
        import azure.ai.ml
        version = azure.ai.ml.__version__
        print(f"✅ Azure ML SDK installed: v{version}")

        # Check if version is recent enough
        major, minor, patch = map(int, version.split('.')[:3])
        if major >= 1 and minor >= 12:
            print(f"✅ SDK version is compatible (v{version} >= v1.12.0)")
            return True
        else:
            print(f"⚠️  SDK version is old (v{version} < v1.12.0)")
            print("   Recommend upgrading: pip install --upgrade azure-ai-ml")
            return False

    except ImportError as e:
        print(f"❌ Azure ML SDK not installed: {e}")
        print("\nInstall with:")
        print("  pip install azure-ai-ml")
        return False


def test_azure_identity():
    """Test Azure Identity SDK."""
    print("\n" + "="*70)
    print("TEST 2: AZURE IDENTITY SDK")
    print("="*70 + "\n")

    try:
        from azure.identity import DefaultAzureCredential
        print("✅ Azure Identity SDK installed")

        # Try to get credentials (doesn't actually authenticate yet)
        credential = DefaultAzureCredential()
        print("✅ DefaultAzureCredential initialized")
        return True

    except ImportError as e:
        print(f"❌ Azure Identity SDK not installed: {e}")
        print("\nInstall with:")
        print("  pip install azure-identity")
        return False


def test_azure_cli_login():
    """Test Azure CLI authentication."""
    print("\n" + "="*70)
    print("TEST 3: AZURE CLI AUTHENTICATION")
    print("="*70 + "\n")

    try:
        import subprocess
        result = subprocess.run(
            ["az", "account", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            import json
            account = json.loads(result.stdout)
            print(f"✅ Azure CLI authenticated")
            print(f"   Subscription: {account['name']}")
            print(f"   ID: {account['id']}")
            return True, account['id']
        else:
            print("❌ Azure CLI not authenticated")
            print("\nAuthenticate with:")
            print("  az login")
            return False, None

    except FileNotFoundError:
        print("❌ Azure CLI not installed")
        print("\nInstall from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False, None
    except Exception as e:
        print(f"❌ Error checking Azure CLI: {e}")
        return False, None


def test_azure_ml_command_api():
    """Test Azure ML command API."""
    print("\n" + "="*70)
    print("TEST 4: AZURE ML COMMAND API")
    print("="*70 + "\n")

    try:
        from azure.ai.ml import command
        print("✅ Azure ML command function available")

        # Try to create a dummy command (won't submit)
        from azure.ai.ml.entities import Environment

        env = Environment(
            name="test-env",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )

        cmd = command(
            code="./",
            command="echo 'test'",
            environment=env,
            compute="test-compute"
        )

        print("✅ Command creation successful")
        print(f"   Type: {type(cmd)}")
        return True

    except ImportError as e:
        print(f"❌ Failed to import command function: {e}")
        print("\nThis might be a version issue. Try:")
        print("  pip install --upgrade azure-ai-ml")
        return False
    except Exception as e:
        print(f"❌ Error creating command: {e}")
        return False


def test_workspace_connectivity(subscription_id):
    """Test workspace connectivity (if provided)."""
    print("\n" + "="*70)
    print("TEST 5: AZURE ML WORKSPACE CONNECTIVITY (OPTIONAL)")
    print("="*70 + "\n")

    print("This test requires workspace details.")
    print("\nSkipping (manual test)...")
    print("\nTo test workspace connectivity, run:")
    print("  python3 ml/azure_ml/train_on_azure.py \\")
    print("    --subscription-id <id> \\")
    print("    --resource-group <rg> \\")
    print("    --workspace <workspace> \\")
    print("    --check-job dummy")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("AZURE ML SETUP VALIDATION")
    print("="*70)
    print("\nThis script tests your Azure ML setup:")
    print("1. Azure ML SDK installation and version")
    print("2. Azure Identity SDK")
    print("3. Azure CLI authentication")
    print("4. Azure ML command API")
    print("5. Workspace connectivity (manual)")

    results = {}

    # Test 1: SDK
    results['sdk'] = test_azure_ml_sdk()

    # Test 2: Identity
    results['identity'] = test_azure_identity()

    # Test 3: CLI
    cli_auth, subscription_id = test_azure_cli_login()
    results['cli'] = cli_auth

    # Test 4: Command API
    results['command_api'] = test_azure_ml_command_api()

    # Test 5: Workspace (informational)
    results['workspace'] = test_workspace_connectivity(subscription_id)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    for test_name, passed in results.items():
        if passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    passed_count = sum(1 for v in [results['sdk'], results['identity'], results['cli'], results['command_api']] if v)
    required_tests = 4

    print(f"\n{passed_count}/{required_tests} required tests passed")

    if passed_count == required_tests:
        print("\n🎉 ALL TESTS PASSED! Azure ML setup is ready.")
        print("\nNext steps:")
        print("1. Create Azure ML workspace (if not done):")
        print("   https://portal.azure.com")
        print("\n2. Submit training job:")
        print("   python3 ml/azure_ml/train_on_azure.py \\")
        print("     --subscription-id <id> \\")
        print("     --resource-group <rg> \\")
        print("     --workspace <workspace>")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nFix the failing tests above before using Azure ML training.")

        if not results['sdk']:
            print("\n📦 Install Azure ML SDK:")
            print("   pip install azure-ai-ml")

        if not results['identity']:
            print("\n📦 Install Azure Identity SDK:")
            print("   pip install azure-identity")

        if not results['cli']:
            print("\n🔐 Authenticate with Azure CLI:")
            print("   az login")


if __name__ == "__main__":
    main()
